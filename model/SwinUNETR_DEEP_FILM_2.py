import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

from transformers import CLIPTextModel, CLIPTokenizer

#FILM based class for adding conditional vector
#this FiLM Block has residual connection along with Layer Norm .
class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, text_dim):
        super().__init__()
        # FiLM parameters
        self.gamma = nn.Linear(text_dim, feature_dim)
        self.beta = nn.Linear(text_dim, feature_dim)
        nn.init.xavier_uniform_(self.gamma.weight)
        nn.init.xavier_uniform_(self.beta.weight)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, feature, text):
        # Store identity for residual
        b, c, d, h, w = feature.shape
        identity = feature
        
        # FiLM modulation
        feature_flat = feature.view(b, c, -1)  # [B, C, D*H*W]
        gamma = self.gamma(text).unsqueeze(-1)
        beta = self.beta(text).unsqueeze(-1)
        modulated = (gamma * feature_flat) + beta
        
        # Apply layer normalization
        # Reshape for layer norm: [B, D*H*W, C]
        modulated = modulated.permute(0, 2, 1)
        modulated = self.layer_norm(modulated)
        # Reshape back: [B, C, D*H*W]
        modulated = modulated.permute(0, 2, 1)
        
        # Reshape and add residual
        output = modulated.view(b, c, d, h, w) + identity
        
        return output

class SwinUNETR_DEEP_FILM(nn.Module):

    def __init__(self,img_size,in_channels,out_channels,feature_size = 48,text_dim = 512,conditonal_vec = True,precomputed_prompt_path = 'dir',model_params=None,normalize = True,spatial_dims=3) -> None:
        super().__init__()
        self.swin_unetr = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=feature_size,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        normalize = normalize,
                        spatial_dims = spatial_dims
                        )

        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        for param in self.text_encoder.parameters():
            param.requires_grad = False


        #adding film block only after the stages of VIT based encoder.
        self.deep_film_block = nn.ModuleList([FiLMBlock(feature_size*(2**(i)),text_dim) for i in range(5)])

        self.default_prompt = "A Computed Tomography of abdomen organ" #yet to figure its use.

        self.class_num = out_channels

        self.normalize = normalize

        self.precomputed_prompts = {}

        if os.path.exists(precomputed_prompt_path):

            dic = pickle.load(open(precomputed_prompt_path,'rb'))
            self.precomputed_prompts = {k:v for k,v in dic.items()}

        if model_params is not None:

            self.load_params(model_params)

    def load_text_emb(self,precomputed_prompt_path):

        if os.path.exists(precomputed_prompt_path):

            dic = pickle.load(open(precomputed_prompt_path,'rb'))
            self.precomputed_prompts = {k:v for k,v in dic.items()}


    def load_params(self,model_dict):

        self.swin_unetr.load_from(model_dict)
        print('Use pretrained weights')

    #check if the whole batch_sent , the keys are present in the precomputed prompt provided dictionary.

    def check_membership(self,batch_keys):
        
        return all(key in self.precomputed_prompts.keys() for key in batch_keys)


    #compute conditoinal vecotrs if the prompts are present or not present
    def compute_conditional(self, prompt):

        device = next(self.parameters()).device
        
        with torch.no_grad():

            if type(prompt) in {list, tuple}:
                # Handle batch of prompts
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
                # Move all tensors to device
                prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
                conditional_vec = self.text_encoder(**prompt_tokens)
                
            elif prompt in self.precomputed_prompts:
                # Handle precomputed prompts
                conditional_vec = self.precomputed_prompts[prompt].float().to(device)
                
            else:
                # Handle single prompt
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                # Move all tensors to device
                prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
                conditional_vec = self.text_encoder(**prompt_tokens)
            
            # Return pooled output
            return conditional_vec['pooler_output'].float()  # get the pooled clip tokens.


    #get the requried text embeddings.
    def get_text_emb(self,prompt=None,batch_size=1):

        if type(prompt)==str:
            cond = self.compute_conditional(prompt)
            cond = cond.repeat(batch_size,1)
        
        elif type(prompt) in {list,tuple} and type(prompt[0])==str:

            assert(len(prompt)==batch_size),"invalid prompt length to batch size"
            #if all the keys lies in the dictionary.
            if self.check_membership(prompt):
                values = [self.precomputed_prompts[key] for key in prompt]
                cond = torch.cat(values, dim=0)

            else:    
                cond = self.compute_conditional(prompt)
        
        elif prompt is not None and type(prompt)==torch.tensor and prompt.ndim==2:
            cond = prompt

        elif prompt is None:

            cond = self.compute_conditional(self.default_prompt)
            cond = cond.repeat(batch_size,1)

        return cond
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, prompt=None):
        # Get text embeddings
        batch_size = x.shape[0]
        text_embs = self.get_text_emb(prompt, batch_size)
        text_embs = text_embs.to(x.device)

        # SwinViT and initial encoder
        hidden_states_out = self.swin_unetr.swinViT(x, self.normalize)
        enc0 = self.swin_unetr.encoder1(x)

        # Encoder path with FiLM conditioning
        enc1 = self.swin_unetr.encoder2(hidden_states_out[0])  #for encoder presenet at stage 0 of VIT.
        film_enc1 = self.deep_film_block[0](enc1, text_embs)

        enc2 = self.swin_unetr.encoder3(hidden_states_out[1]) #for encoder present at stage 1 of vit 
        film_enc2 = self.deep_film_block[1](enc2, text_embs)

        enc3 = self.swin_unetr.encoder4(hidden_states_out[2]) #for encoder present at stage 2 of VIT
        film_enc3 = self.deep_film_block[2](enc3, text_embs)

        film_enc4 = self.deep_film_block[3](hidden_states_out[3], text_embs)  #since HS3 doesn't pass through any encoder in orignal architecture. no encoder at stage 3 of VIT. 

        dec4 = self.swin_unetr.encoder10(hidden_states_out[4]) #after encoder at stage 4 of VIT. 
        film_dec4 = self.deep_film_block[4](dec4, text_embs)  # Enc10 output through FilM4

        # Decoder path
        dec3 = self.swin_unetr.decoder5(film_dec4, film_enc4)  
        dec2 = self.swin_unetr.decoder4(dec3, film_enc3)
        dec1 = self.swin_unetr.decoder3(dec2, film_enc2)
        dec0 = self.swin_unetr.decoder2(dec1, film_enc1)
        out = self.swin_unetr.decoder1(dec0, enc0)
        
        logits = self.swin_unetr.out(out)
        return logits
