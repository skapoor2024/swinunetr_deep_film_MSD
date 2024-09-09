import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetOutBlock

from transformers import CLIPTextModel, CLIPTokenizer

#FILM based class for adding conditional vector

class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, text_dim):
        super().__init__()
        self.gamma = nn.Linear(text_dim, feature_dim)
        nn.init.xavier_uniform_(self.gamma.weight)  # Initialize gamma weights with Xavier uniform
        self.beta = nn.Linear(text_dim, feature_dim)
        nn.init.xavier_uniform_(self.beta.weight)  # Initialize beta weights with Xavier uniform


    def forward(self, feature, text):
        b, c, d, h, w = feature.shape
        feature_flat = feature.view(b, c, -1) #shape(B,C,H*W*D)
        #print(text.dtype)
        gamma = self.gamma(text).unsqueeze(-1) #shape(B,C,1)
        beta = self.beta(text).unsqueeze(-1) #shape(B,C,1) this is broadcastable.
        
        modulated = (gamma * feature_flat) + beta
        return modulated.view(b, c, d, h, w)

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

            
        # bottleneck_features = feature_size*16 #bottleneck feature size at [-1] of VIT

        # self.film_block = FiLMBlock(bottleneck_features,text_dim)

        #adding film block only after the stages of VIT based encoder.
        self.deep_film_block = nn.ModuleList([FiLMBlock(feature_size*(2**(i+1)),text_dim) for i in range(4)])

        self.default_prompt = "A Computed Tomography of abdomen organ" #yet to figure its use.

        self.class_num = out_channels

        self.normalize = normalize

        self.precomputed_prompts = {}

        if os.path.exists(precomputed_prompt_path):

            dic = pickle.load(open(precomputed_prompt_path,'rb'))
            self.precomputed_prompts = {k:v for k,v in dic.items()}

        if model_params is not None:

            self.load_params(model_params)


    def load_params(self,model_dict):

        self.swin_unetr.load_from(model_dict)
        print('Use pretrained weights')

    #check if the whole batch_sent , the keys are present in the precomputed prompt provided dictionary.

    def check_membership(self,batch_keys):
        
        return all(key in self.precomputed_prompts.keys() for key in batch_keys)


    #compute conditoinal vecotrs if the prompts are present or not present
    def compute_conditional(self,prompt):

        device = next(self.parameters()).device

        with torch.no_grad():

            if type(prompt) in {list,tuple}:

                prompt_tokens = self.tokenizer(prompt).to(device)
                conditional_vec = self.text_encoder(prompt_tokens).float().to(device)
            
            elif prompt in self.precomputed_prompts:

                conditional_vec = self.precomputed_prompts[prompt].float().to(device)
            
            else:

                prompt_tokens = self.tokenizer([prompt]).to(device)
                conditional_vec = self.text_encoder(prompt_tokens)[0].float().to(device)
        
        return conditional_vec['pooler_output'] #get the pooled clip tokens.


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


    def forward(self,x,prompt = None,return_features = False):
        
        batch_size = x.shape[0]
        text_embs = self.get_text_emb(prompt,batch_size)
        text_embs = text_embs.to(x.device)
        #print(x.device)

        hidden_states_out = self.swin_unetr.swinViT(x, self.normalize)
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states_out[0])
        enc2 = self.swin_unetr.encoder3(hidden_states_out[1])
        enc3 = self.swin_unetr.encoder4(hidden_states_out[2])
        dec4 = self.swin_unetr.encoder10(hidden_states_out[4])
         # print(x_in.shape, enc0.shape, enc1.shape, enc2.shape, enc3.shape, dec4.shape)
        # torch.Size([6, 1, 64, 64, 64]) torch.Size([6, 48, 64, 64, 64]) torch.Size([6, 48, 32, 32, 32]) 
        # torch.Size([6, 96, 16, 16, 16]) torch.Size([6, 192, 8,8, 8]) torch.Size([6, 768, 2, 2, 2])

        #adding film based embeddings after residual layer in the bottleneck
        film_bottleneck_4 = self.deep_film_block[3](dec4,text_embs)
        dec3 = self.swin_unetr.decoder5(film_bottleneck_4, hidden_states_out[3])

        film_bottleneck_3 = self.deep_film_block[2](dec3,text_embs)
        dec2 = self.swin_unetr.decoder4(film_bottleneck_3, enc3)

        film_bottleneck_2 = self.deep_film_block[1](dec2,text_embs)
        dec1 = self.swin_unetr.decoder3(film_bottleneck_2, enc2)

        film_bottleneck_1 = self.deep_film_block[0](dec1,text_embs)
        dec0 = self.swin_unetr.decoder2(film_bottleneck_1, enc1)

        out = self.swin_unetr.decoder1(dec0, enc0)
        logits = self.swin_unetr.out(out)
        return logits
        # print(dec3.shape, dec2.shape, dec1.shape, dec0.shape, out.shape)
        # torch.Size([6, 384, 4, 4, 4]) torch.Size([6, 192, 8, 8, 8]) torch.Size([6, 96, 16, 16, 16]) 
        # torch.Size([6, 48, 32, 32, 32]) torch.Size([6, 48, 64, 64, 64])
