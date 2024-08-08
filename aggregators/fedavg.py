import copy
import torch

class Aggregator(torch.nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def aggregate(self, agents, resource_manager, device):
        resource = resource_manager.find_resource({"task": "aggregation"})
        print(f"Aggregating models on resource {resource}")

        global_model = {}

        for name, _ in agents[0].get_model_params().items():
            global_model[name] = torch.stack([agent.get_model_params()[name].to(device) for agent in agents]).mean(0)
        
        for agent in agents:
            agent.set_model_params({name: param.to(agent.device) for name, param in global_model.items()})
            

def fed_avg(w, args):
    w_avg = {key: torch.zeros_like(w[0][key]).to(args.device) for key in w[0]}
        for i in range(len(w)):
        for key in w_avg:
            w_avg[key] += w[i][key]
    for key in w_avg:
        w_avg[key] /= len(w)
        
    return {key: torch.tensor(val).to(args.device) for key, val in w_avg.items()}
