from RangeAgent import EvalAgentDeepRange
from PokerRL.game.games import StandardLeduc  # or any other game

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

import numpy as np
import torch
import torch.nn as nn
import pickle

import time
from copy import deepcopy

t_prof = TrainingProfile(
    name="DEEP_RANGE_v0",
    nn_type="feedforward",
    
    max_buffer_size_adv=3e6,
    eval_agent_export_freq=20,  # export API to play against the agent
    n_traversals_per_iter=1500,
    n_batches_adv_training=750,
    n_batches_avrg_training=2000,
    n_merge_and_table_layer_units_adv=64,
    n_merge_and_table_layer_units_avrg=64,
    n_units_final_adv=64,
    n_units_final_avrg=64,
    mini_batch_size_adv=2048,
    mini_batch_size_avrg=2048,
    init_adv_model="last",
    init_avrg_model="last",
    use_pre_layers_adv=False,
    use_pre_layers_avrg=False,

    game_cls=StandardLeduc,

    # You can specify one or both modes. Choosing both is useful to compare them.
    eval_modes_of_algo=(
     EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
     EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,  # Deep CFR
    ),

    DISTRIBUTED=False,
)

range_loss = nn.CrossEntropyLoss()
action_loss = nn.MSELoss() ## cross-entropy would be ideal

def hole_card_onehot(hole_card):
    rank = hole_card[0][0]
    suit = hole_card[0][1]
    out = rank + suit * 3 ## arbitrary but it will learn the relationship
    return torch.LongTensor([out])

def distill(student_agent, teacher_agent, args={'lr':1e-2, 'iters': 10000, 'lambda': 10}):
    """
    Distill student_agent to play like teacher_agent
    """
    
    def loss(range1, range2, action1, action2):
        return range_loss(range1, range2) + action_loss(action1, action2)
        
    env_bldr = student_agent.env_bldr
    env_cls = env_bldr.env_cls
    env_args = env_bldr.env_args
    lut_holder = env_cls.get_lut_holder()
    
    assert(student_agent.env_bldr.env_cls == teacher_agent.env_bldr.env_cls)
    assert(env_args.n_seats == 2)

    optimizer = torch.optim.Adam(list(student_agent.policy._net.parameters()), lr=args['lr'])
    start_time = time.time()
    
    REFERENCE_AGENT = 0
    
    _env = env_cls(env_args=env_args, lut_holder=lut_holder, is_evaluating=True)
    _eval_agents = [teacher_agent, deepcopy(teacher_agent)]
    
    results = {
        "range_loss": [],
        "action_loss": [],
        "total_loss": [],
    }
    iters = 0 # number of hands played
    evals = 0 # number of teaching moments
    
    # zero grads, set net to train mode
    student_agent.policy._net.train()
    optimizer.zero_grad()

    while iters < args['iters']:
        iters += 1
        
        if iters % 200 == 0:
            print("Iters {} | Evals {} | RangeLoss {} | ActionLoss {} | TotalLoss {}".format(
                iters, evals, sum(results['range_loss']) / evals, sum(results['action_loss']) / evals, sum(results['total_loss']) / evals
            ))
            
            # print("gradient:", list(student_agent.policy._net.parameters())[0].grad)

            # print("old params:", list(student_agent.policy._net.parameters())[0])

            optimizer.step()

            # print("new params:", list(student_agent.policy._net.parameters())[0])
            
            optimizer.zero_grad()

        
        for seat_p0 in range(_env.N_SEATS):
            seat_p1 = 1 - seat_p0
            
            # """""""""""""""""
            # Reset Episode
            # """""""""""""""""
            _, r_for_all, done, info = _env.reset()
            for e in _eval_agents + [student_agent]:
                e.reset(deck_state_dict=_env.cards_state_dict())

            # """""""""""""""""
            # Play Episode
            # """""""""""""""""

            while not done:
                p_id_acting = _env.current_player.seat_id

                if p_id_acting == seat_p0:
                    evals += 1 #increment counter
                    
                    # set student to position of agent 1, estimate range + actions
                    student_agent.set_env_wrapper(_eval_agents[REFERENCE_AGENT]._internal_env_wrapper) 
                    student_a_probs = student_agent.get_a_probs_tensor()
                    student_range_probs = student_agent.get_range_probs()
                    
                    # get true values 
                    a_probs = torch.Tensor(_eval_agents[REFERENCE_AGENT].get_a_probs())
                    range_label = _env.get_hole_cards_of_player(seat_p1) #get opponent's true range
                    range_label = hole_card_onehot(range_label) # convert to label
                    action_int, _ = _eval_agents[REFERENCE_AGENT].get_action(step_env=True, need_probs=False)
                    
                    # print("True:", a_probs, range_label)
                    # print("Prediction:", student_a_probs, student_range_probs)
                    # print("Checking requires_grad:", student_a_probs.requires_grad, student_range_probs.requires_grad)
                    
                    # compute loss
                    rloss = range_loss(student_range_probs.view(1,-1), range_label)
                    aloss = action_loss(student_a_probs, a_probs)
                    loss = rloss + args['lambda'] * aloss
                    
                    results['total_loss'].append(loss)
                    results['range_loss'].append(rloss)
                    results['action_loss'].append(aloss)
                    
                    # print("Loss:", rloss, aloss, loss)
                    
                    # backpropogate
                    loss.backward() # accumulate gradients over many steps
                                        
                    # notify opponent
                    _eval_agents[1 - REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                       action_he_did=action_int)
                elif p_id_acting == seat_p1:
                    a_probs = _eval_agents[REFERENCE_AGENT].get_a_probs()
                    action_int, _ = _eval_agents[1 - REFERENCE_AGENT].get_action(step_env=True, need_probs=False)
                    _eval_agents[REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                   action_he_did=action_int)
                else:
                    raise ValueError("Only HU supported!")
                
                _, r_for_all, done, info = _env.step(action_int)  
    
    end_time = time.time()
    print("Time taken", end_time - start_time)

    print(optimizer)
    
    return results

agent_file1 = "/home/leduc/poker_ai_data/eval_agent/SD-CFR_LEDUC_EXAMPLE_200/120/eval_agentAVRG_NET.pkl"

student_agent = EvalAgentDeepRange(t_prof, mode=None, device=None)
teacher_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=agent_file1)

results = distill(student_agent, teacher_agent, args={'lr':1e-2, 'iters': 500000, 'lambda': 1})
name = "deep_range_500000_1"

student_agent.save_to_file(name + ".pt")

pickle.dump(results, open(name + "_log.pkl", "wb" ))
