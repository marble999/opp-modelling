import torch
import torch.nn as nn

from PokerRL.game._.rl_env.game_rules import HoldemRules, LeducRules

class RangeNet(nn.Module):

    def __init__(self, range_net_args, env_bldr, device):
        super().__init__()
        self.args = range_net_args
        self.env_bldr = env_bldr
        self.n_cards = LeducRules.N_CARDS_IN_DECK # TO-DO: change this eventually to fit more games than just StandardLeduc
        self.n_hands = self.n_cards # * (self.n_cards+1) / 2 (for two cards)

        MPM = range_net_args.mpm_args.get_mpm_cls()

        self._relu = nn.ReLU(inplace=False)
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=self.args.mpm_args)

        self._final_layer = nn.Linear(in_features=self._mpm.output_units, out_features=self.args.n_units_final)
        self._out_layer = nn.Linear(in_features=self.args.n_units_final, out_features=self.n_hands)
        self._softmax = nn.Softmax(dim=1)

        self.to(device)

    def forward(self, pub_obses, range_idxs):

        out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        out = self._relu(self._final_layer(out))
        out = self._out_layer(out)
        return self._softmax(out) #this IS softmaxed

class RangeActionNet(nn.Module):

    def __init__(self, range_net_args, env_bldr, device):
        super().__init__()
        self.args = range_net_args
        self.env_bldr = env_bldr
        self.n_actions = self.env_bldr.N_ACTIONS
        
        # self.range_net = RangeNet(range_net_args, env_bldr, device)
        # self._relu = nn.ReLU(inplace=False)
        # self._final_layer = nn.Linear(in_features=27, out_features=self.args.n_units_final) #TO-DO: change this 
        # self._out_layer = nn.Linear(in_features=self.args.n_units_final, out_features=self.env_bldr.N_ACTIONS)
        
        MPM = range_net_args.mpm_args.get_mpm_cls()
        self._relu = nn.ReLU(inplace=False)
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=self.args.mpm_args)
        self._final_layer = nn.Linear(in_features=self._mpm.output_units, out_features=self.args.n_units_final)
        self._out_layer = nn.Linear(in_features=self.args.n_units_final, out_features=self.n_actions)

        self.to(device)

    def forward(self, pub_obses, range_idxs, legal_action_masks, hand_info):
        """
        Softmax is not applied in here! It is separate in training and action fns
        """
        mode = 'regular' # 'range'
        
        if mode == 'range':
            ranges = self.range_net(pub_obses, range_idxs)
            out = torch.cat([ranges, torch.Tensor(hand_info).view(1,-1)], dim=1)
            out = self._relu(self._final_layer(out))
            out = self._out_layer(out)
            out = torch.where(legal_action_masks == 1,
                              out,
                              torch.FloatTensor([-10e20]).to(device=out.device).expand_as(out))
            return out
        
        else:
            out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
            out = self._relu(self._final_layer(out))
            out = self._out_layer(out)
            out = torch.where(legal_action_masks == 1,
                              out,
                              torch.FloatTensor([-10e20]).to(device=out.device).expand_as(out))
            return out

class RangeNetArgs:

    def __init__(self,
                 mpm_args,
                 n_units_final
                 ):
        self.mpm_args = mpm_args
        self.n_units_final = n_units_final
