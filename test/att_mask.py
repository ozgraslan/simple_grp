import  torch

def get_att_mask(mask_arr, inp_mask):
    cumsum = torch.cumsum(mask_arr, dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = inp_mask[:, None, :] * inp_mask[:, :, None]
    ## need to add the diagonal to the mask
    ## because older torch versions give nans 
    ## if a token does not attend to anything
    return attn_mask * valid_mask + torch.eye(valid_mask.shape[1]).unsqueeze(0)

inp_mask = torch.tensor([[1] + [0] + [0] + [1] + [1] + [1] + [1]])

att_mask = get_att_mask(torch.tensor([[1] + [0] + [0] + [0] + [0] + [1] + [1]]), inp_mask).bool()
print(att_mask)
print(torch.eye(7).shape)
