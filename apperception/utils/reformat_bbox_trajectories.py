def reformat_bbox_trajectories(bbox_trajectories):
    result = {}
    for meta in bbox_trajectories:
        item_id, coordinates, timestamp = meta[0], meta[1:-1], meta[-1]
        if item_id in result:
            result[item_id][0].append(coordinates)
            result[item_id][1].append(timestamp)
        else:
            result[item_id] = [[coordinates], [timestamp]]

    return result
