import argparse
import math
import random

# from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import repeat

# import tqdm
from tqdm.contrib.concurrent import process_map
from sympy.functions.combinatorial.factorials import factorial, subfactorial

from permumark.ecc import ReedSolomonCode
from permumark.permutation_mapping import PermutationMapping
from attacks import make_permutation


def simulate_corruption(pm, gf_size, group_size, perm_type, strategy):
    digit = random.randint(0, gf_size - 1)
    pi = pm.encode(digit, "attention")
    if group_size == 1:
        sigma = make_permutation(len(pi), perm_type).tolist()
        pi_ = [pi[s] for s in sigma]
    else:
        kv_base_perm, group_perms = pi

        kv_base_perm_ = kv_base_perm[:]
        if strategy == "intra" or strategy == "both":
            sigma = make_permutation(len(kv_base_perm), perm_type).tolist()
            kv_base_perm_ = [kv_base_perm[s] for s in sigma]

        group_perms_ = [group_perm[:] for group_perm in group_perms]
        if strategy == "inter" or strategy == "both":
            idx = random.randint(0, len(group_perms) - 1)
            sigma = make_permutation(len(group_perms[0]), perm_type).tolist()
            group_perms_[idx] = [group_perms_[idx][s] for s in sigma]

        pi_ = (kv_base_perm_, group_perms_)

    extracted_pi = pm.decode(pi_, "attention")

    return extracted_pi is not None


def estimate_prob(num: int, gf_size: int, num_kv_heads: int, group_size: int = 1):
    print(
        f"Probability estimation setting: num_kv_heads: {num_kv_heads}, "
        f"group_size: {group_size}, gf_size: {gf_size}, repeat: {num}"
    )

    pm = PermutationMapping(gf_size, 1024, 1024, num_kv_heads, group_size)

    size = int(factorial(num_kv_heads))
    if group_size > 1:
        size *= int(factorial(group_size) ** num_kv_heads)
    baseline1 = gf_size / size

    size = int(subfactorial(num_kv_heads))
    if group_size > 1:
        size *= int(subfactorial(group_size) ** num_kv_heads)
    baseline2 = gf_size / size

    final_results = {}
    perm_types = ["random", "swap", "shift", "derangement"]
    strategies = ["intra"]
    if group_size > 1:
        strategies.extend(["inter", "both"])
    for strategy in strategies:
        for perm_type in perm_types:
            undetected_count = 0
            for i in range(0, num, 10**7):
                sub_num = min(10**7, num - i)
                partial_results = process_map(
                    simulate_corruption,
                    repeat(pm, sub_num),
                    repeat(gf_size, sub_num),
                    repeat(group_size, sub_num),
                    repeat(perm_type, sub_num),
                    repeat(strategy, sub_num),
                    chunksize=2000,
                    desc=f"{strategy:>5}, {perm_type:>11} ({i//10**7})",
                )
                undetected_count += sum(partial_results)

            p = undetected_count / num
            final_results[(strategy, perm_type)] = p

    print(f"Theoretical baselines: {baseline1:.3e}, {baseline2:.3e}")
    for strategy in strategies:
        for perm_type in perm_types:
            p = final_results[(strategy, perm_type)]
            se = math.sqrt(p * (1.0 - p) / num)
            print(
                f"{strategy:>5}, {perm_type:>11}: {p:.3e} ± {se:.3e} (95% confidence)"
            )

    return baseline1, baseline2


def simulate_forgery(
    gf_size: int,
    n: int,
    original_perms,
    slots: list[str],
    pm: PermutationMapping,
    ecc: ReedSolomonCode,
    budget: int,
    perm_type: str,
):
    perms = deepcopy(original_perms)
    # select slots to attack
    slots_to_attack = [("attention", i) for i in range(min(budget, n // 2))]
    budget -= min(budget, n // 2)
    if budget > 0:
        slots_to_attack.append(("embeddings", -1))
        budget -= 1
    slots_to_attack.extend([("feed_forward", i) for i in range(budget)])

    for slot_type, index in slots_to_attack:
        if slot_type == "embeddings":
            perm = perms[0]
            sigma = make_permutation(len(perm), perm_type)
            perms[0] = [perm[s] for s in sigma]
        elif slot_type == "attention":
            perm = perms[2 * index + 1]
            if isinstance(perm, tuple):
                base_perm, group_perms = perm
                sigma = make_permutation(len(base_perm), perm_type)
                base_perm = [base_perm[s] for s in sigma]
                for i in range(len(group_perms)):
                    sigma = make_permutation(len(group_perms[i]), perm_type)
                    group_perms[i] = [group_perms[i][s] for s in sigma]
                perms[2 * index + 1] = (base_perm, group_perms)
            else:
                sigma = make_permutation(len(perm), perm_type)
                perms[2 * index + 1] = [perm[s] for s in sigma]
        else:
            perm = perms[2 * index + 2]
            sigma = make_permutation(len(perm), perm_type)
            perms[2 * index + 2] = [perm[s] for s in sigma]

    decoded_watermark = list(pm.decode_perms(perms, slots))
    decoded_watermark_ = [0] * len(decoded_watermark)
    erasure_idx = []
    for i, digit in enumerate(decoded_watermark):
        if digit is not None and 0 <= digit < gf_size:
            decoded_watermark_[i] = digit
        else:
            erasure_idx.append(i)
    decoded_identity = ecc.decode(decoded_watermark_, erasure_idx)

    return decoded_identity, decoded_watermark_, erasure_idx


def batch_simulate_forgery(
    batch_size: int,
    identity: list[int],
    watermark: list[int],
    gf_size: int,
    n: int,
    original_perms,
    slots: list[str],
    pm: PermutationMapping,
    ecc: ReedSolomonCode,
    budget: int,
    perm_type: str,
):
    # 1. number of corruption
    # 2. number of erasure (detected corruption)
    # 3. decoded to None
    # 4. decoded to different identity
    total_corruption_num = 0
    total_erasure_num = 0
    total_decode_failure_num = 0
    total_forgery_success_num = 0

    for i in range(batch_size):
        decoded_identity, decoded_watermark, erasure_idx = simulate_forgery(
            gf_size, n, original_perms, slots, pm, ecc, budget, perm_type
        )
        forgery_success = (decoded_identity is not None) and (
            decoded_identity != identity
        )
        erasure_num = len(erasure_idx)
        corruption_num = sum(d1 != d2 for d1, d2 in zip(decoded_watermark, watermark))
        total_corruption_num += corruption_num
        total_erasure_num += erasure_num
        total_decode_failure_num += decoded_identity is None
        total_forgery_success_num += forgery_success
    return (
        total_corruption_num,
        total_erasure_num,
        total_decode_failure_num,
        total_forgery_success_num,
    )


def estimate_forgery_prob(
    num: int,
    budget: int,
    gf_size: int,
    k: int,
    n: int,
    hidden_size: int,
    num_kv_heads: int,
    group_size: int,
    intermediate_size: int,
    perm_type: str,
):
    assert budget > 0, f"budget must be positive: {budget}"
    assert budget <= n, f"budget must be less than {n}: {budget}"
    pm = PermutationMapping(
        gf_size, hidden_size, intermediate_size, num_kv_heads, group_size
    )
    ecc = ReedSolomonCode(gf_size, n, k)
    identity = [random.randint(0, gf_size - 1) for _ in range(k)]
    watermark = ecc.encode(identity)
    slots = list(("embeddings",) + ("attention", "feed_forward") * (n // 2))
    perms = list(pm.encode_codeword(watermark, slots))

    # batch_size = 2
    #
    # total_corruption_num = 0
    # total_erasure_num = 0
    # total_decode_failure_num = 0
    # total_forgery_success_num = 0
    #
    # futures = []
    # with ProcessPoolExecutor() as executor:
    #     for _ in range(0, num, batch_size):
    #         futures.append(
    #             executor.submit(
    #                 batch_simulate_forgery,
    #                 batch_size,
    #                 identity,
    #                 watermark,
    #                 gf_size,
    #                 n,
    #                 perms,
    #                 slots,
    #                 pm,
    #                 ecc,
    #                 budget,
    #                 perm_type,
    #             )
    #         )
    #     print(f"Submitted {num} batches")
    #     for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
    #         corruption_num, erasure_num, decode_failure_num, forgery_success_num = future.result()
    #         total_corruption_num += corruption_num
    #         total_erasure_num += erasure_num
    #         total_decode_failure_num += decode_failure_num
    #         total_forgery_success_num += forgery_success_num
    #
    # return (
    #     total_corruption_num,
    #     total_erasure_num,
    #     total_decode_failure_num,
    #     total_forgery_success_num,
    # )

    forgery_success_list = []
    erasure_num_list = []
    corruption_num_list = []

    results = process_map(
        simulate_forgery,
        repeat(gf_size),
        repeat(n, times=num),
        repeat(perms),
        repeat(slots),
        repeat(pm),
        repeat(ecc),
        repeat(budget),
        repeat(perm_type),
        total=num,
        chunksize=10,
    )
    for decoded_identity, decoded_watermark, erasure_idx in results:
        forgery_success = (decoded_identity is not None) and (
            decoded_identity != identity
        )
        erasure_num = len(erasure_idx)
        corruption_num = sum(d1 != d2 for d1, d2 in zip(decoded_watermark, watermark))
        forgery_success_list.append(forgery_success)
        erasure_num_list.append(erasure_num)
        corruption_num_list.append(corruption_num)

    total_forgery_success = sum(forgery_success_list)
    total_undetected_corruption_num = sum(
        x - y for x, y in zip(corruption_num_list, erasure_num_list)
    )

    return total_forgery_success, total_undetected_corruption_num


def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate the probability of forgery and undetected corruption"
    )
    parser.add_argument(
        "task", type=str, choices=["corruption", "forgery"], help="Task type"
    )
    parser.add_argument("--num", type=int, default=10**3, help="Number of simulations")
    parser.add_argument("--gf_size", type=int, default=2**24, help="Galois field size")
    parser.add_argument("--k", type=int, default=1, help="Number of identity digits")
    parser.add_argument("--n", type=int, default=33, help="Reed-Solomon code length")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden size")
    parser.add_argument(
        "--num_kv_heads", type=int, default=8, help="Number of KV heads"
    )
    parser.add_argument("--group_size", type=int, default=4, help="Group size")
    parser.add_argument(
        "--intermediate_size", type=int, default=8192, help="Intermediate size"
    )
    parser.add_argument(
        "--perm_type", type=str, default="shift", help="Type of permutation"
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.task == "corruption":
        estimate_prob(
            args.num,
            args.gf_size,
            args.num_kv_heads,
            args.group_size,
        )
    elif args.task == "forgery":
        forgery_success_num, undetected_corruption_num = estimate_forgery_prob(
            args.num,
            args.n,
            args.gf_size,
            args.k,
            args.n,
            args.hidden_size,
            args.num_kv_heads,
            args.group_size,
            args.intermediate_size,
            args.perm_type,
        )
        print(f"Forgery success: {forgery_success_num}")
        print(f"Undetected corruption: {undetected_corruption_num}")
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
