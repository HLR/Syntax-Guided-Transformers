import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model):
        """
        Inputs
            max_len - Maximum length of a sequence to expect.
            d_model - Hidden dimensionality of the input.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x

class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class GatedRelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds model capacity {self.max_len}"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = torch.nn.LeakyReLU(attn)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class BertEmbeddings(nn.Module):
    """Create embeddings from word, position embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.l_hidden_size
        )
        
        if config.pos_embed == 'learned':
            self.pos_embed = 'learned'
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.l_hidden_size
            )
        elif config.pos_embed == 'sincos':
            self.pos_embed = 'sincos'
            self.position_embeddings = PositionalEncoding(
                config.max_position_embeddings, config.l_hidden_size
            )
        elif config.pos_embed == 'relative':
            self.pos_embed = 'relative'
            self.position_embeddings = RelativePositionalEncoding(
                config.max_position_embeddings, config.l_hidden_size,
                config.rpe_heads, config.rpe_dropout
            )

    
        self.LayerNorm = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.l_hidden_dropout_prob)
        print(f"language in pos embedding {self.pos_embed}")        

    def forward(self, input_ids, position_ids=None):

        words_embeddings = self.word_embeddings(input_ids)
        
        if self.pos_embed == 'learned':
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = self.position_embeddings(words_embeddings)


        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    

class BertOutputEmbeddings(nn.Module):
    """Create embeddings from word, position embeddings.
    """

    def __init__(self, config):
        super(BertOutputEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.target_vocab_size, config.l_hidden_size
        )
        
        if config.pos_embed == 'learned':
            self.pos_embed = 'learned'
            self.position_embeddings = nn.Embedding(
                config.target_max_position_embeddings, config.l_hidden_size
            )
        elif config.pos_embed == 'sincos':
            self.pos_embed = 'sincos'
            self.position_embeddings = PositionalEncoding(
                config.target_max_position_embeddings, config.l_hidden_size
            )
        elif config.pos_embed == 'relative':
            self.pos_embed = 'relative'
            self.position_embeddings = RelativePositionalEncoding(
                config.target_max_position_embeddings, config.l_hidden_size,
                config.rpe_heads, config.rpe_dropout
            )

        self.LayerNorm = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.l_hidden_dropout_prob)
        print(f"ouptut pos embedding {self.pos_embed}")        

    def forward(self, input_ids, position_ids=None):
        
        words_embeddings = self.word_embeddings(input_ids)
        if self.pos_embed == 'learned':
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = self.position_embeddings(words_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertImageEmbeddings(nn.Module):
    """Create embeddings from image, spatial location.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)

        if config.v_pos_embed == 'linear':
            self.pos_embed = 'linear'
            self.image_location_embeddings = nn.Linear(config.v_loc_size, config.v_hidden_size)
        elif config.v_pos_embed == 'learned':
            self.pos_embed = 'learned'
            self.position_embeddings = nn.Embedding(
                37, config.v_hidden_size
            )
        elif config.v_pos_embed == 'sincos':
            self.pos_embed = 'sincos'
            self.position_embeddings = PositionalEncoding(
                37, config.v_hidden_size
            )
        elif config.v_pos_embed == 'relative':
            self.pos_embed = 'relative'
            self.position_embeddings = RelativePositionalEncoding(
                37, config.v_hidden_size,
                config.v_rpe_heads, config.v_rpe_dropout
            )
        
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

        print(f"vis pos embedding {self.pos_embed}")        

    def forward(self, input_ids, input_loc):
        
        img_embeddings = self.image_embeddings(input_ids)
        if self.pos_embed == 'linear':
            loc_embeddings = self.image_location_embeddings(input_loc)
            embeddings = img_embeddings + loc_embeddings
        elif self.pos_embed == 'learned':
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings = img_embeddings + position_embeddings
        elif self.pos_embed == 'sincos':
            embeddings = self.position_embeddings(img_embeddings)
        elif self.pos_embed == 'relative':
            embeddings = self.position_embeddings(img_embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    
    
class SimpleImageEmbeddings(nn.Module):
    """Create embeddings from image, spatial location.
    """

    def __init__(self, config):
        super(SimpleImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.simple_embed_size)
        self.image_location_embeddings = nn.Linear(config.v_loc_size, config.simple_embed_size)
        
    def forward(self, input_ids, input_loc):

        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = img_embeddings + loc_embeddings
        
        return embeddings
    
    
class SimpleEmbeddings(nn.Module):
    """Create embeddings from word, position embeddings.
    """

    def __init__(self, config):
        super(SimpleEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.simple_embed_size
        )
        
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.simple_embed_size
        )

    def forward(self, input_ids, position_ids=None):

        words_embeddings = self.word_embeddings(input_ids)
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        return embeddings