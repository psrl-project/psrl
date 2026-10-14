#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import torch
from nixl._api import nixl_agent, nixl_agent_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=29876)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # initiator use default port
    listen_port = args.port
    if args.mode != "target":
        listen_port = 0

    if args.cuda >= 0:
        torch.set_default_device(f"cuda:{args.cuda}")
    else:  # To be sure this is the default
        torch.set_default_device("cpu")

    config = nixl_agent_config(True, True, listen_port)

    # Allocate memory and register with NIXL
    agent = nixl_agent(args.mode, config)

    # for k in sorted(os.environ):
    #    print(f"{k}={os.environ[k]}")

    if args.mode == "target":
        # 0.01GiB, 1000 tensors
        tensors = [torch.zeros(1024**3 // 2 // 128, dtype=torch.bfloat16) for i in range(1000)]
    else:
        tensors = [torch.ones(1024**3 // 2 // 128, dtype=torch.bfloat16) for i in range(1000)]

    # print(f"{args.mode} Tensors: {tensors}")

    start_time = time.time()
    reg_descs = agent.register_memory(tensors)
    end_time = time.time()
    print(f"Register memory time: {end_time - start_time} seconds")
    if not reg_descs:  # Same as reg_descs if successful
        print("Memory registration failed.")
        exit()

    # Target code
    if args.mode == "target":
        ready = False

        target_descs = reg_descs.trim()
        target_desc_str = agent.get_serialized_descs(target_descs)

        # Send desc list to initiator when metadata is ready
        while not ready:
            ready = agent.check_remote_metadata("initiator")

        agent.send_notif("initiator", target_desc_str)

        print("Waiting for transfer")

        # The completion payload is a UUID here, though arbitrary Python bytes are supported.
        while not agent.check_remote_xfer_done("initiator", b"UUID"):
            continue

    # Initiator code
    else:
        print("Initiator sending to " + args.ip)
        agent.fetch_remote_metadata("target", args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)

        notifs = agent.get_new_notifs()

        while len(notifs) == 0:
            notifs = agent.get_new_notifs()

        target_descs = agent.deserialize_descs(notifs["target"][0])
        initiator_descs = reg_descs.trim()
        print(f"Initiator descs count: {initiator_descs.descCount()}")

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            ready = agent.check_remote_metadata("target")

        print("Ready for transfer")

        start_time = time.time()
        xfer_handle = agent.initialize_xfer("WRITE", initiator_descs, target_descs, "target", "UUID")
        end_time = time.time()
        print(f"Initialize transfer time: {end_time - start_time} seconds")

        if not xfer_handle:
            print("Creating transfer failed.")
            exit()

        start_time = time.time()
        state = agent.transfer(xfer_handle)
        end_time = time.time()
        print(f"Post transfer time: {end_time - start_time} seconds")

        if state == "ERR":
            print("Posting transfer failed.")
            exit()

        start_time = time.time()
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                break
        end_time = time.time()
        print(f"Wait transfer time: {end_time - start_time} seconds")

    # Verify data after read
    for i, tensor in enumerate(tensors):
        if not torch.allclose(tensor, torch.ones(1024**3 // 2 // 128, dtype=torch.bfloat16)):
            print(f"Data verification failed for tensor {i}: {tensor}")
            exit()
    if args.mode != "target":
        agent.remove_remote_agent("target")
        agent.release_xfer_handle(xfer_handle)
        agent.invalidate_local_metadata(args.ip, args.port)

    agent.deregister_memory(reg_descs)

    print("Test Complete.")
