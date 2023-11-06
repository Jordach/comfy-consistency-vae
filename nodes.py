from .consistencydecoder import ConsistencyDecoder
from PIL import Image
import torch
import numpy as np
import os

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

pwd = os.getcwd()

class Consistency:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required":
				{
					"latent": ("LATENT",)
				}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "decode"
	CATEGORY = "latent"

	def decode(self, latent):
		#print(latent)
		decoder_consistency = ConsistencyDecoder(device="cuda:0", download_root=pwd)
		consistent_latent = decoder_consistency(latent["samples"].to("cuda:0"))
		del decoder_consistency
		image = consistent_latent[0].cpu().numpy()
		image = (image + 1.0) * 127.5
		image = image.clip(0, 255).astype(np.uint8)
		image = Image.fromarray(image.transpose(1, 2, 0))
		return conv_pil_tensor(image)


NODE_CLASS_MAPPINGS = {
	"Comfy_ConsistencyVAE": Consistency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"Comfy_ConsistencyVAE": "Consistency VAE Decoder",
}