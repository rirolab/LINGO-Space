import torch
from torch.distributions import Normal, VonMises


def cartesian2polar(coords):
    """Convert Cartesian coordinates to polar coordinates.
    Args:
        coords (torch.Tensor): Cartesian coordinates of shape (*, 2).
    
    Returns:
        torch.Tensor: Polar coordinates of shape (*, 2).
    """
    x, y = coords[..., 0], coords[..., 1]
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    return torch.stack((r, theta), dim=-1)

def plot_spatial_distribution(parameters, pos, width=640, height=320, max_mean=None):
    """
    Plot the spatial distribution of the parameters
    Args:
        parameters (list): List of parameters of shape (num_nodes, 5)
        pos (torch.Tensor): Position of shape (num_nodes, 2)
        width (int): Width of the image
        height (int): Height of the image
    Returns:
        Z (torch.Tensor): Spatial distribution of shape (num_nodes, width, height)
    """
    if width > height:
        x_start = -1
        x_end = 1
        y_start = -height / width
        y_end = height / width
    else:
        x_start = -width / height
        x_end = width / height
        y_start = -1
        y_end = 1
        
    x = torch.linspace(x_start, x_end, width, device=pos.device)
    y = torch.linspace(y_start, y_end, height, device=pos.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.zeros(len(parameters), width, height, device=pos.device)
    for i, output in enumerate(parameters):
        num_nodes = output.size(0)
        for n in range(num_nodes):
            log_w,  mean, var, loc, con = output[n]
            # print(n, torch.exp(log_w), mean)
            if max_mean is not None:
                mean = min(mean, max_mean)
            gaussian = Normal(mean, torch.sqrt(var + 1e-9))
            vonmises = VonMises(loc, con)
            polar_coord = cartesian2polar(torch.stack([X, Y], dim=-1) - pos[n])
            gaussian_log_prob = gaussian.log_prob(polar_coord[..., 0])
            vonmises_log_prob = vonmises.log_prob(polar_coord[..., 1])
            log_prob = gaussian_log_prob + vonmises_log_prob + log_w
            Z[i] += torch.exp(log_prob)
    return Z, (X, Y)

def get_max_point(parameters, pos, width=640, height=320):
    """
    Get the maximum point of the spatial distribution
    Args:
        parameters (list): List of parameters of shape (num_nodes, 5)
        pos (torch.Tensor): Position of shape (num_nodes, 2)
        width (int): Width of the image
        height (int): Height of the image
    Returns:
        r (int): Row index of the maximum point
        c (int): Column index of the maximum point
    """
    Z , _= plot_spatial_distribution(parameters, pos, width, height)
        
    finalZ = torch.prod(Z, dim=0)
    r, c = torch.where(finalZ == torch.max(finalZ))
    r = r[0].item()
    c = c[0].item()
    return r, c

def check_spatial_relation(location, predicate, ref_obj_bbox):
    """
    Check spatial relation between the location and the reference object
    Args:
        location (torch.Tensor): Location of shape (2,) in pixel coordinate
        predicate (str): Spatial relation predicate
        ref_obj_bbox (torch.Tensor): Bounding box of the reference object of shape (4,) in pixel coordinate
    Returns:
        bool: True if the location satisfies the spatial relation predicate    
    """

    if predicate == 'left':
        return location[0].item() < ref_obj_bbox[0].item()
    
    elif predicate == 'right':
        return location[0].item() > ref_obj_bbox[2].item()
    
    elif predicate == 'above':
        return location[1].item() < ref_obj_bbox[1].item()
    
    elif predicate == 'below':
        return location[1].item() > ref_obj_bbox[3].item()
    
    elif predicate == 'in' or predicate == 'center':
        c1 = location[0].item() > ref_obj_bbox[0].item()
        c2 = location[0].item() < ref_obj_bbox[2].item()
        c3 = location[1].item() > ref_obj_bbox[1].item()
        c4 = location[1].item() < ref_obj_bbox[3].item()
        return c1 and c2 and c3 and c4
    
    elif predicate == 'below left' or predicate == 'left below':
        return check_spatial_relation(location, 'below', ref_obj_bbox) and \
            check_spatial_relation(location, 'left', ref_obj_bbox)
    
    elif predicate == 'below right' or predicate == 'right below':
        return check_spatial_relation(location, 'below', ref_obj_bbox) and \
            check_spatial_relation(location, 'right', ref_obj_bbox)
    
    elif predicate == 'above left' or predicate == 'left above':
        return check_spatial_relation(location, 'above', ref_obj_bbox) and \
            check_spatial_relation(location, 'left', ref_obj_bbox)
    
    elif predicate == 'above right' or predicate == 'right above':
        return check_spatial_relation(location, 'above', ref_obj_bbox) and \
            check_spatial_relation(location, 'right', ref_obj_bbox)
    
    elif 'close' in predicate:
        bbox_center = (ref_obj_bbox[:2] + ref_obj_bbox[2:]) / 2
        return torch.norm(bbox_center - location).item() < 150
    
    elif 'far' in predicate:
        bbox_center = (ref_obj_bbox[:2] + ref_obj_bbox[2:]) / 2
        return torch.norm(bbox_center - location).item() > 300

    else:
        raise ValueError(f'Unknown predicate: {predicate}')
