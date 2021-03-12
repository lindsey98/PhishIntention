import numpy as np
from .iou import bbox_boarder_dist_simple

def nearby_check(boarder_dist_x, boarder_dist_y, direction='both', nearby_ts=50., remove_diag=False):
    '''
    Find nearby elements
    :param boarder_dist_x: NxM boarder distance matrix in horizontal direction
    :param boarder_dist_y: NxM boarder distance matrix in vertical direction
    :param direction: find nearby elements in x/y/both direction(s)
    :param nearby_ts: nearby/not nearby threshold in pixels
    :param remove_diag: ignore diagnal nearby elements (enable when comparing with itself)
    :return nearby: NxM boolean matrix specifying pairwise neighboring relations
    '''
    assert direction in ['both', 'x', 'y']
    # check if there are any nearby elements in vertical/horizontal directions
    nearby_x = np.multiply(np.equal(boarder_dist_y, np.zeros_like(boarder_dist_y)).astype('float'), 
                           np.less_equal(boarder_dist_x, np.ones_like(boarder_dist_x)*nearby_ts).astype('float'))     

    nearby_y = np.multiply(np.equal(boarder_dist_x, np.zeros_like(boarder_dist_x)).astype('float'), 
                           np.less_equal(boarder_dist_y, np.ones_like(boarder_dist_y)*nearby_ts).astype('float'))

    if remove_diag:
        # remove diagonal -- diagonal is the distance w.r.t. itself 
        nearby_x = np.multiply(nearby_x, 1-np.eye(nearby_x.shape[0]))
        nearby_y = np.multiply(nearby_y, 1-np.eye(nearby_y.shape[0]))
        
    if direction == 'both':
        nearby = np.logical_or(nearby_x.astype(bool), nearby_y.astype(bool)).astype(bool)
    elif direction == 'x':
        nearby = nearby_x.astype(bool)
    elif direction == 'y':
        nearby = nearby_y.astype(bool)
        
    return nearby
        
def layout_heuristic(pred_boxes, pred_classes):
    '''
    Define simple heuristic for identifying suspicious layout
    :param pred_boxes: Nx4 bbox coords
    :param pred_classes: Nx1 bbox types 
    :return pattern_ct: how many times a specific pattern can be found 
    '''
    
    # look at inputs, labels, buttons
    inputs, buttons, labels = pred_boxes[pred_classes==1], pred_boxes[pred_classes==2], pred_boxes[pred_classes==3] 
    
    # convert to numpy if not
    inputs = inputs.numpy() if not isinstance(inputs, np.ndarray) else inputs
    buttons = buttons.numpy() if not isinstance(buttons, np.ndarray) else buttons
    labels = labels.numpy() if not isinstance(labels, np.ndarray) else labels

    
    # find label input * >=2
    # pattern
    pattern_ct_1 = 0
    if len(labels) <= 1 or len(inputs) <= 1: # no box/too few boxes
        pass
    else:
        boarder_dist_x, boarder_dist_y = bbox_boarder_dist_simple(labels, inputs)
        assert boarder_dist_x.shape == boarder_dist_y.shape
        
        # check if there are any nearby elements in vertical/horizontal directions
        nearby = nearby_check(boarder_dist_x, boarder_dist_y)
#         print('Nearby matrix:\n', nearby.astype('float'))
        
        # if either is True --> nearby element
        pattern_ct_1 = np.sum(nearby.astype('float'))

    # find input * >=2
    #      button
    # pattern  
    pattern_ct_2 = 0
    if len(inputs) <= 1 or len(buttons) == 0: # no box/too few boxes
        pass
    else:    
        boarder_dist_x, boarder_dist_y = bbox_boarder_dist_simple(inputs, inputs)
        
        nearby_1 = nearby_check(boarder_dist_x=boarder_dist_x, boarder_dist_y=boarder_dist_y, 
                                direction='y', remove_diag=True)
        
#         print('Nearby input:\n', nearby_1.astype('float'))
        # update pattern_ct
        pattern_ct_2 = np.sum(nearby_1.astype('float'))
        
        if pattern_ct_2 >= 3:
            pass
        
        # further check whether there is a button below input
        elif pattern_ct_2 >= 2:
            boarder_dist_x_2, boarder_dist_y_2 = bbox_boarder_dist_simple(inputs, buttons)
            
            nearby_2 = nearby_check(boarder_dist_x=boarder_dist_x_2, boarder_dist_y=boarder_dist_y_2, 
                                    direction='y')            
#             print('Nearby button around input:\n', nearby_2.astype('float'))
            # There is no button below input: revert back to no pattern
            if np.sum(nearby_2.astype('float')) == 0:
                pattern_ct_2 = 0
    
    # take the maximum
    return max(pattern_ct_1, pattern_ct_2), len(inputs)