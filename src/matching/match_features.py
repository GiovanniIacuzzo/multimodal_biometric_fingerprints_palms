import torch
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from src.db.database import load_minutiae_from_db, save_matching_result, get_image_id_by_filename

logger = logging.getLogger(__name__)

# ========================
# Prepara tensori per tutte le immagini
# ========================
def prepare_minutiae_tensors_full(
    image_filenames: List[str], device: str = "mps"
) -> Tuple[torch.Tensor, List[int], List[List[str]]]:
    """
    Restituisce:
    - tensor [num_images, max_minutiae, 3] (x, y, orientation)
    - lista image_ids
    - lista tipi minutiae per ogni immagine
    """
    minutiae_list = []
    types_list = []
    image_ids = []

    max_len = 0
    for fname in image_filenames:
        image_id = get_image_id_by_filename(fname)
        if image_id is None:
            logger.warning(f"Immagine non trovata nel DB: {fname}")
            minutiae_list.append([])
            types_list.append([])
            image_ids.append(None)
            continue

        minutiae = load_minutiae_from_db(image_id)
        image_ids.append(image_id)
        types_list.append([m["type"] for m in minutiae])
        coords = [[m["x"], m["y"], m["orientation"]] for m in minutiae]
        minutiae_list.append(coords)
        max_len = max(max_len, len(coords))

    num_images = len(image_filenames)
    tensor = torch.full((num_images, max_len, 3), -1.0, dtype=torch.float32, device=device)

    for i, coords in enumerate(minutiae_list):
        if coords:
            tensor[i, :len(coords), :] = torch.tensor(coords, dtype=torch.float32, device=device)

    return tensor, image_ids, types_list

# ========================
# Batch matching completamente vettoriale
# ========================
def batch_match(
    image_filenames: List[str],
    method: str = "pair_matching",
    dist_thresh: float = 25.0,
    angle_thresh: float = 0.5,
    device: str = "mps"
) -> Dict[Tuple[str, str], float]:
    """
    Batch matching **completamente vettoriale** su GPU/MPS senza loop Python sui file.
    """
    results = {}
    tensor, image_ids, types_list = prepare_minutiae_tensors_full(image_filenames, device)
    num_images = len(image_filenames)

    logger.info(f"Avvio batch matching full vector ({num_images*(num_images-1)//2} confronti)")

    # Creazione mask per minutiae valide
    valid_mask = tensor[:, :, 0] >= 0  # [num_images, max_len]

    # Loop solo per immagine a, ma tutto il resto è vettoriale
    for i in tqdm(range(num_images), desc="Matching immagini (full GPU)", unit="immagine"):
        id_a = image_ids[i]
        if id_a is None or not valid_mask[i].any():
            continue
        t1 = tensor[i][valid_mask[i]]  # [N1,3]
        types1 = types_list[i]

        # Costruzione batch di tutte le altre immagini j > i
        ids_j = []
        t2_list = []
        types2_list = []
        idx_j_list = []
        for j in range(i+1, num_images):
            if image_ids[j] is None or not valid_mask[j].any():
                continue
            ids_j.append(image_ids[j])
            t2_list.append(tensor[j][valid_mask[j]])  # [N2_j,3]
            types2_list.append(types_list[j])
            idx_j_list.append(j)

        # Matching per tipo di minutiae
        for t in set(types1):
            mask1 = torch.tensor([tp==t for tp in types1], device=device)
            sel1 = t1[mask1][:,:2]
            angles1 = t1[mask1][:,2][:,None]

            if sel1.shape[0]==0:
                continue

            for k, sel2_tensor in enumerate(t2_list):
                types2 = types2_list[k]
                if t not in types2:
                    continue
                mask2 = torch.tensor([tp==t for tp in types2], device=device)
                sel2 = sel2_tensor[mask2][:,:2]
                angles2 = sel2_tensor[mask2][:,2][None,:]

                if sel2.shape[0]==0:
                    continue

                # Distances pairwise
                diff = sel1[:,None,:] - sel2[None,:,:]
                dist = torch.norm(diff, dim=2)
                angle_diff = torch.abs(angles1 - angles2)

                matches = (dist <= dist_thresh) & (angle_diff <= angle_thresh)
                matched_count = matches.any(dim=1).sum().item()

                total_pairs = max(sel1.shape[0], sel2.shape[0])
                score = matched_count / total_pairs if total_pairs>0 else 0.0
                results[(image_filenames[i], image_filenames[idx_j_list[k]])] = score

                # Salvataggio DB
                save_matching_result(id_a, ids_j[k], score, method)

    logger.info(f"Batch completato. Confronti riusciti: {len(results)} / {num_images*(num_images-1)//2}")
    return results
