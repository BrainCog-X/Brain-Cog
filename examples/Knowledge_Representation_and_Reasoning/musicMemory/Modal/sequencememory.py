from .synapse import Synapse


class SequenceMemory():
    '''
    classdocs
    '''

    def __init__(self, neutype):
        '''
        Constructor
        '''
        self.neutype = neutype
        self.sequenceLayers = {}

    def createActionSequenceMem(self, layernum, neutype, neunumpergroup):
        pass

    def doRemembering(self):
        pass

    def doConnecting(self, goal, sl, order):
        # the goal and the group always generate spikes in a limit time window,create a synapse between them.
        group = sl.groups.get(order)
        if (group == None): return
        tb = (order - 1) * group.timeWindow
        te = (order) * group.timeWindow
        sp1_goal = {}
        sp2 = []

        for n in goal.neurons:
            sp = []
            for st in n.spiketime:
                if (st < te and st >= tb):
                    sp.append(st)
            sp1_goal[n.index] = sp

        for n in group.neurons:
            if (len(n.spiketime) > 0):
                for index, sp in sp1_goal.items():
                    temp = 0
                    for sp1 in n.spiketime:  # spike times of group
                        for sp2 in sp:
                            if (abs(sp1 - sp2) <= n.tau_ref):
                                temp += 1
                    if (
                            temp >= 4):  # super threshold, create a new synapse between goal and neurons of sequence group
                        syn = Synapse(goal.neurons[index - 1], n)
                        syn.type = 2
                        syn.weight = 3
                        n.pre_neurons.append(goal.neurons[index - 1])
                        n.synapses.append(syn)

                        # add reverse synapse to neurons of the goal
                        syn2 = Synapse(n, goal.neurons[index - 1])
                        syn2.type = 2
                        syn2.weight = 1
                        goal.neurons[index - 1].synapses.append(syn2)

        # clear the goal's spike time

    ''' *************************************************************
    I have forgot why neurons here needs to be cleaned, but this must be important, mark here
    for n in goal.neurons:
        n.spiketime = []    
    ******************************************************************
    '''

    #     def doConnectToGoal(self,goal,track,order): # connect to the goal in the time window
    #
    #             for sl in track.values():
    #                 group = sl.groups.get(order)
    #
    #                 if(group == None): continue
    #                 # the goal and the group always generate spikes in a limit time window,create a synapse between them.
    #                 tb = (order-1)*group.timeWindow
    #                 te = (order) * group.timeWindow
    #                 sp1_goal = {}
    #                 sp2 = []
    #
    #                 for n in goal.neurons:
    #                     sp = []
    #                     for st in n.spiketime:
    #                         if(st < te and st >= tb ):
    #                             sp.append(st)
    #                     sp1_goal[n.index] = sp
    #
    #                 for n in group.neurons:
    #                     if(len(n.spiketime) > 0):
    #                         for index,sp in sp1_goal.items():
    #                             temp = 0
    #                             for sp1 in n.spiketime: #spike times of group
    #                                 for sp2 in sp:
    #                                     if(abs(sp1-sp2) <= n.tau_ref):
    #                                         temp += 1
    #                             if(temp >= 4): # super threshold, create a new synapse between goal and neurons of sequence group
    #                                 syn = Synapse(goal.neurons[index-1],n)
    #                                 syn.type = 2
    #                                 syn.weight = 3
    #                                 n.pre_neurons.append(goal.neurons[index-1])
    #                                 n.synapses.append(syn)
    #
    #                                 #add reverse synapse to neurons of the goal
    #                                 syn2 = Synapse(n,goal.neurons[index-1])
    #                                 syn2.type = 2
    #                                 syn2.weight = 1
    #                                 goal.neurons[index-1].synapses.append(syn2)
    #
    #             #clear the goal's spike time
    #             ''' *************************************************************
    #             I have forgot why neurons here needs to be cleaned, but this must be important, mark here
    #             for n in goal.neurons:
    #                 n.spiketime = []
    #             ******************************************************************
    #         '''
    #
    #     def doConnectToComposer(self, composer, track, order):
    #         for sl in track.values():
    #             group = sl.groups.get(order)
    #
    #             if(group == None): continue
    #             # the goal and the group always generate spikes in a limit time window,create a synapse between them.
    #             tb = (order-1)*group.timeWindow
    #             te = (order) * group.timeWindow
    #             sp1_composer = {}
    #             sp2 = []
    #
    #             for n in composer.neurons:
    #                 sp = []
    #                 for st in n.spiketime:
    #                     if(st < te and st >= tb ):
    #                         sp.append(st)
    #                 sp1_composer[n.index] = sp
    #
    #             for n in group.neurons:
    #                 if(len(n.spiketime) > 0):
    #                     for index,sp in sp1_composer.items():
    #                         temp = 0
    #                         for sp1 in n.spiketime: #spike times of group
    #                             for sp2 in sp:
    #                                 if(abs(sp1-sp2) <= n.tau_ref):
    #                                     temp += 1
    #                         if(temp >= 4): # super threshold, create a new synapse between composer and neurons of sequence group
    #                             syn = Synapse(composer.neurons[index-1],n)
    #                             syn.type = 2
    #                             syn.weight = 3
    #                             n.pre_neurons.append(composer.neurons[index-1])
    #                             n.synapses.append(syn)
    #
    #                             #add reverse synapse to neurons of the goal
    # #                             syn2 = Synapse(n,goal.neurons[index-1])
    # #                             syn2.type = 2
    # #                             syn2.weight = 1
    # #                             goal.neurons[index-1].synapses.append(syn2)
    #
    #
    #     def doConnectToGenre(self, genre, track, order):
    #         for sl in track.values():
    #             group = sl.groups.get(order)
    #
    #             if(group == None): continue
    #             # the goal and the group always generate spikes in a limit time window,create a synapse between them.
    #             tb = (order-1)*group.timeWindow
    #             te = (order) * group.timeWindow
    #             sp1_genre = {}
    #             sp2 = []
    #
    #             for n in genre.neurons:
    #                 sp = []
    #                 for st in n.spiketime:
    #                     if(st < te and st >= tb ):
    #                         sp.append(st)
    #                 sp1_genre[n.index] = sp
    #
    #             for n in group.neurons:
    #                 if(len(n.spiketime) > 0):
    #                     for index,sp in sp1_genre.items():
    #                         temp = 0
    #                         for sp1 in n.spiketime: #spike times of group
    #                             for sp2 in sp:
    #                                 if(abs(sp1-sp2) <= n.tau_ref):
    #                                     temp += 1
    #                         if(temp >= 4): # super threshold, create a new synapse between composer and neurons of sequence group
    #                             syn = Synapse(genre.neurons[index-1],n)
    #                             syn.type = 2
    #                             syn.weight = 3
    #                             n.pre_neurons.append(genre.neurons[index-1])
    #                             n.synapses.append(syn)

    def setTestStates(self):
        for itrack in self.sequenceLayers.values():
            for sl in itrack.values():
                sl.setTestStates()