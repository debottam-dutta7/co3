import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from PIL import Image
import numpy as np
from composers.Co3 import Co3
from composers.utils_custom import seed_everything
import json
# import spacy
import stanza
import pyrallis
from composers.config import Co3Config
from composers.utils.ptp_utils import get_prompts_and_concepts_coarse


def get_result_path(opt, prefix=None)-> str:
    if prefix is None:
        prefix = f"./output/test/{opt.prompt}_{opt.seed}"
    if not prefix.endswith('/'):
        prefix = prefix + '/'
    if opt.num_ts_to_correct > 0:
        prefix += f"algo-{opt.corrector_algo}_"
    else:
        prefix += f"algo-sd"
    return prefix

def get_prompts_and_concepts_string(config, nlp):
    '''
    config.prompt_orig: full prompt
    config.prompt: NOUN chunks w/o adjectives + full_prompt
    config.concepts: NOUN chunks w/ adjectives + full prompt
    returns
    ------
    prompts = [full_prompt] + <NOUN chunks w/o adjectives>
    concepts =  <NOUN chunks w/ adjectives> + background (which is currently set to full prompt)
    prompts_single: should be used in the MoLE stage later. 
    '''
    prompt_orig = config.prompt_orig 

    prompts, concepts = get_prompts_and_concepts_coarse(prompt_orig, nlp,
                                                 remove_adj_from_contrastive_prompts=True,
                                                 remove_art_from_contrastive_prompts=True) 
    concept_bg = prompt_orig
    prompt = "+".join(prompts) + f"+{prompt_orig}"
    concepts = concepts + [concept_bg]
    concept = "+".join(concepts)
    prompt_single = concept

    return prompt, concept, prompt_single 

def prepare_prompts(config):
    prompt_sep = config.prompt.split('+')
    concepts = config.concepts.split('+')
    prompts = []
    prompts.append(config.prompt_orig)
    concept_num = len(concepts)
    config.concept_num = concept_num
    config.prompts_single = concepts
    config.prompt = prompts + prompt_sep
    return config


@pyrallis.wrap()
def main(opt: Co3Config):

    ae_prompt_path = opt.promptset_path
    with open(ae_prompt_path, 'r') as f:
        prompt_dict = json.load(f)

    prompt_categories = list(prompt_dict.keys())
    root_output_path = get_result_path(opt, prefix=opt.output_path)

    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path, exist_ok=True)

    co3 = Co3(opt)
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', download_method=None)

    for category in prompt_categories:
        prompts = prompt_dict[category]

        seeds = opt.seeds
        for seed in seeds:
            for i, prompt in enumerate(prompts):
                print(f"Generating {i+1}/{len(prompts)} for seed {seed} in category {category}")

                opt.seed = seed
                opt.prompt_orig = prompt

                save_dir = os.path.join(root_output_path, category, "samples")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prompt}_{opt.seed}.png")
                prompt_str, concept, _ = get_prompts_and_concepts_string(opt, nlp)
                opt.prompt = prompt_str
                opt.concept = concept
                seed_everything(opt.seed)
                co3.config = opt
                co3.prepare_prompts(opt)
                co3.prepare_embeds()
                img = co3.run_sampling()
                img[0].save(save_path)
                print(f"Saved: {save_path}")

    

if __name__ == "__main__":
    main()