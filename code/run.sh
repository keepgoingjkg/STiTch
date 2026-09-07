datapath=data_path
python main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B/32
python main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L/14
python main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-bigG-14

datapath=data_path
python main.py --dataset fashioniq_dress --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B/32
python main.py --dataset fashioniq_dress --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L/14
python main.py --dataset fashioniq_dress --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-bigG-14

datapath=data_path
python main.py --dataset fashioniq_toptee --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B/32
python main.py --dataset fashioniq_toptee --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L/14
python main.py --dataset fashioniq_toptee --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-bigG-14

