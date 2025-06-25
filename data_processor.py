import minari
import random
import tqdm

datasets = ['Minimal-Hopper-Expert-v5', 'Minimal-Walker2d-Expert-v5', 'Minimal-HalfCheetah-Expert-v5',
            'Hopper-Expert-v5', 'Walker2d-Expert-v5', 'HalfCheetah-Expert-v5']

for dataset_name in datasets:
    dataset = minari.load_dataset(dataset_name)
    num_observations = dataset.spec.observation_space.shape[0]
    num_actions = dataset.spec.action_space.shape[0]
    
    with open(f"{dataset_name}", 'w') as f:
        f.write(";; ")
        for i in range(num_observations):
            f.write(f"OBS{i+1} ")
        for i in range(num_actions):
            f.write(f"ACTION{i+1} ")
        f.write("REWARD ")
        f.write("TERM ")
        f.write("TRUNC")
        f.write("\n")
        
        f.write("(")
        for episode in tqdm.tqdm(dataset, desc=f"Processing {dataset_name}"):
            
            for (observation, action, reward, term, trunc) in zip(episode.observations, episode.actions, episode.rewards, episode.terminations, episode.truncations):
                f.write('(')
                f.write(' '.join(str(x) for x in observation))
                f.write(' ')
                f.write(' '.join(str(x) for x in action))
                f.write(' ')
                f.write(str(reward))
                f.write(" ")
                f.write('T' if term else 'NIL')
                f.write(" ")
                f.write('T' if trunc else 'NIL')
                f.write(')')
                
        f.write(")\n")