#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pickle
import time

import torch
from nixl._api import nixl_agent, nixl_agent_config
from transformers import AutoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=23457)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Hugging Face model directory",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy mode: create tensors with same shapes as model but without loading actual weights",
    )
    parser.add_argument(
        "--save_tensor_shapes",
        type=str,
        default=None,
        help="Path to save tensor shapes to a file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # initiator use default port
    listen_port = args.port
    if args.mode != "target":
        listen_port = 0

    config = nixl_agent_config(True, True, listen_port)

    device = torch.device("cuda:0" if args.cuda >= 0 else "cpu")

    # Allocate memory and register with NIXL
    agent = nixl_agent(args.mode, config)

    # Load Hugging Face model or create dummy tensors
    if args.dummy:
        print(f"Dummy mode: Loading model metadata from {args.model_path}")
        if not os.path.exists(args.model_path):
            print(f"Model path {args.model_path} does not exist!")
            exit(1)

        # Load model with meta device to get shapes without loading weights
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="meta")
        state_dict = model.state_dict()

        print(f"Model metadata loaded with {len(state_dict)} parameters")

        # Create dummy tensors with same shapes as the model
        tensors = []
        tensor_names = []
        tensor_shapes = []

        for name, tensor in state_dict.items():
            # Create dummy tensor with same shape and dtype
            dummy_tensor = torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)
            tensors.append(dummy_tensor)
            tensor_names.append(name)
            tensor_shapes.append(tensor.shape)

        if args.save_tensor_shapes:
            with open(args.save_tensor_shapes, "wb") as f:
                pickle.dump(tensor_shapes, f)

        print(f"Created {len(tensors)} dummy tensors with same shapes as model")

        # Print some tensor shape information
        total_elements = sum(tensor.numel() for tensor in tensors)
        print(f"Total elements across all tensors: {total_elements:,}")
        print(f"Sample tensor shapes: {[tensor.shape for tensor in tensors[:5]]}")

        # Set all weights to 1 on initiator side for verification
        if args.mode != "target":
            print("Setting all dummy weights to 1 on initiator side...")
            for tensor in tensors:
                tensor.fill_(1.0)
            print("All dummy weights set to 1")
    else:
        print(f"Loading model from {args.model_path}")
        if not os.path.exists(args.model_path):
            print(f"Model path {args.model_path} does not exist!")
            exit(1)

        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=device)
        state_dict = model.state_dict()

        print(f"Model loaded with {len(state_dict)} parameters")

        # Convert state dict to list of tensors
        tensors = []
        tensor_names = []
        for name, tensor in state_dict.items():
            tensors.append(tensor)
            tensor_names.append(name)

        print(f"Total tensors to transfer: {len(tensors)}, device: {tensors[0].device}")

        # Set all weights to 1 on initiator side for verification
        if args.mode != "target":
            print("Setting all weights to 1 on initiator side...")
            for tensor in tensors:
                tensor.fill_(1.0)
            print("All weights set to 1")

    # Register each tensor separately
    print(f"Registering {len(tensors)} tensors individually...")
    reg_descs_list = []
    total_register_time = 0

    for i, tensor in enumerate(tensors):
        start_time = time.time()
        reg_desc = agent.register_memory([tensor])
        end_time = time.time()
        register_time = end_time - start_time
        total_register_time += register_time

        if not reg_desc:
            print(f"Memory registration failed for tensor {i} ({tensor_names[i]})")
            exit()

        reg_descs_list.append(reg_desc)
        print(f"Tensor {i} ({tensor_names[i]}) registered in {register_time:.4f} seconds")

    print(f"Total register memory time: {total_register_time:.4f} seconds")
    print(f"Average register time per tensor: {total_register_time / len(tensors):.4f} seconds")

    # Target code
    if args.mode == "target":
        ready = False

        # Process each tensor descriptor separately
        target_descs_list = []
        for i, reg_desc in enumerate(reg_descs_list):
            target_desc = reg_desc.trim()
            target_descs_list.append(target_desc)

        # Serialize all descriptors
        target_desc_strs = []
        for i, target_desc in enumerate(target_descs_list):
            desc_str = agent.get_serialized_descs(target_desc)
            target_desc_strs.append(desc_str)

        # Send desc list to initiator when metadata is ready
        while not ready:
            ready = agent.check_remote_metadata("initiator")

        # Send all descriptor strings
        for i, desc_str in enumerate(target_desc_strs):
            agent.send_notif("initiator", desc_str)

        print("Waiting for transfer")

        # Waiting for transfer to complete
        while True:
            if agent.check_remote_xfer_done("initiator", b"MODEL_TRANSFER"):
                break

        print("Model transfer completed")

    # Initiator code
    else:
        print("Initiator sending to " + args.ip)
        agent.fetch_remote_metadata("target", args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)

        # Wait for target descriptors (now we expect multiple notifications)
        target_descs_list = []
        while len(target_descs_list) < len(tensors):
            notifs = agent.get_new_notifs()
            if len(notifs) != 0 and "target" in notifs:
                target_descs_list += notifs["target"]
            print(f"Received {len(target_descs_list)}/{len(tensors)} target descriptors")

        # Deserialize all target descriptors
        target_descs_list = [agent.deserialize_descs(desc_str) for desc_str in target_descs_list]

        # Process each tensor separately
        initiator_descs_list = []
        for i, reg_desc in enumerate(reg_descs_list):
            initiator_desc = reg_desc.trim()
            initiator_descs_list.append(initiator_desc)

        print(f"Target descriptors count: {sum(desc.descCount() for desc in target_descs_list)}")
        print(f"Initiator descriptors count: {sum(desc.descCount() for desc in initiator_descs_list)}")

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            ready = agent.check_remote_metadata("target")

        print("Ready for transfer")

        # Transfer each tensor separately
        print("Starting transfer of all model tensors individually...")
        total_start_time = time.time()

        xfer_handles = []
        total_initialize_time = 0
        total_post_time = 0

        for i in range(len(tensors)):
            print(f"Processing tensor {i} ({tensor_names[i]})...")

            # Initialize transfer for this tensor
            start_time = time.time()
            xfer_handle = agent.initialize_xfer(
                "READ",
                initiator_descs_list[i],
                target_descs_list[i],
                "target",
                "MODEL_TRANSFER",
            )
            end_time = time.time()
            initialize_time = end_time - start_time
            total_initialize_time += initialize_time

            if not xfer_handle:
                print(f"Creating transfer failed for tensor {i} ({tensor_names[i]})")
                exit()

            # Perform transfer
            start_time = time.time()
            state = agent.transfer(xfer_handle)
            end_time = time.time()
            post_time = end_time - start_time
            total_post_time += post_time

            if state == "ERR":
                print(f"Posting transfer failed for tensor {i} ({tensor_names[i]})")
                exit()

            xfer_handles.append(xfer_handle)
            print(f"Tensor {i} ({tensor_names[i]}): Initialize: {initialize_time:.4f}s, Post: {post_time:.4f}s")

        print(f"Total initialize transfer time: {total_initialize_time:.4f} seconds")
        print(f"Total post transfer time: {total_post_time:.4f} seconds")

        # Wait for all transfers to complete
        print("Waiting for all transfers to complete...")
        start_time = time.time()
        for i, xfer_handle in enumerate(xfer_handles):
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    print(f"Transfer got to Error state for tensor {i} ({tensor_names[i]})")
                    exit()
                elif state == "DONE":
                    break
        end_time = time.time()
        wait_time = end_time - start_time
        print(f"Wait for all transfers to complete time: {wait_time:.4f} seconds")

        total_end_time = time.time()
        total_transfer_time = total_end_time - total_start_time
        print(f"Total transfer time: {total_transfer_time:.4f} seconds")

        # Clean up all transfer handles
        for i, xfer_handle in enumerate(xfer_handles):
            agent.release_xfer_handle(xfer_handle)

    # Verify data after read (only for target mode)
    if args.mode == "target":
        print("Verifying transferred data...")
        verification_passed = True
        for i, (name, tensor) in enumerate(zip(tensor_names, tensors)):
            # For verification, we expect all tensors to be 1 (since initiator set them to 1)
            expected_tensor = torch.ones_like(tensor)
            if not torch.allclose(tensor, expected_tensor, rtol=1e-5, atol=1e-8):
                print(f"Data verification failed for tensor {i} ({name})")
                print(f"Expected all values to be 1, but got min={tensor.min():.6f}, max={tensor.max():.6f}")
                verification_passed = False
                break

        if verification_passed:
            if args.dummy:
                print("Data verification passed. All dummy tensors are 1.")
            else:
                print("Data verification passed. All tensors are 1.")
        else:
            print("Data verification failed")
            exit(1)

    if args.mode != "target":
        agent.remove_remote_agent("target")
        agent.invalidate_local_metadata(args.ip, args.port)

    # Deregister all memory
    for i, reg_desc in enumerate(reg_descs_list):
        agent.deregister_memory(reg_desc)

    print("Test Complete.")
