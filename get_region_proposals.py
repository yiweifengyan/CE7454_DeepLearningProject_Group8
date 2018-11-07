import selectivesearch

def get_region_proposal (img, scale=450, sigma=0.6, min_size=5):
    # perform selective search0-
    img_lbl, regions = selectivesearch.selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        if r['size'] < 100:
            continue
        if r['rect'][2] > 360 * 3 / 4 or r['rect'][3] > 640 * 3 / 4:
            continue
        if r['rect'][2] < 3 or r['rect'][3] < 3:
            continue
        candidates.add(r['rect'])
    return candidates

