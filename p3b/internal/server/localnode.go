package server

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
)

// LocalNodeClient reaches a node agent in this process.
//
// A single-machine deployment runs control and node together, so the control
// plane calls the node directly rather than over a socket to itself. The
// interface is the same either way, which is what lets a fleet swap in a gRPC
// client without the control plane knowing.
type LocalNodeClient struct {
	node *Node
}

// NewLocalNodeClient returns a client for a node in this process.
func NewLocalNodeClient(node *Node) *LocalNodeClient {
	return &LocalNodeClient{node: node}
}

// Admit asks the node to accept one request against its live state.
func (l *LocalNodeClient) Admit(ctx context.Context, nodeID string, spec backend.Spec) (string, []int32, string, error) {
	resp, err := l.node.Admit(ctx, &v1.AdmitRequest{Spec: specToProto(spec)})
	if err != nil {
		return "", nil, "", err
	}
	return resp.GetLeaseId(), resp.GetGpuIndices(), resp.GetRefusal(), nil
}

// CreateOn provisions against an admission the node already granted.
func (l *LocalNodeClient) CreateOn(
	ctx context.Context, nodeID, leaseID, backendName string, spec backend.Spec, callback string,
) (backend.Created, error) {
	resp, err := l.node.CreateOn(ctx, &v1.CreateOnRequest{
		LeaseId: leaseID, Spec: specToProto(spec), CallbackTarget: callback, Backend: backendName,
	})
	if err != nil {
		return backend.Created{}, err
	}
	return backend.Created{
		Handle: backend.Handle{
			Backend:   resp.GetHandle().GetBackend(),
			SandboxID: resp.GetHandle().GetSandboxId(),
			NodeID:    resp.GetHandle().GetNodeId(),
		},
		Capabilities: capabilitiesFromProto(resp.GetCapabilities()),
		Agent: backend.AgentEndpoint{
			Address:           resp.GetAgent().GetAddress(),
			Headers:           resp.GetAgent().GetHeaders(),
			CallbackHostAlias: resp.GetAgent().GetCallbackHostAlias(),
			CallbackPort:      resp.GetAgent().GetCallbackPort(),
		},
		WarmStart: resp.GetWarmStart(),
	}, nil
}

// ReleaseOn destroys one sandbox the node holds.
func (l *LocalNodeClient) ReleaseOn(ctx context.Context, handle backend.Handle) error {
	_, err := l.node.ReleaseOn(ctx, &v1.SandboxHandle{
		Backend: handle.Backend, SandboxId: handle.SandboxID, NodeId: handle.NodeID,
	})
	return err
}

// StatusOn reports one sandbox's state from the node that holds it.
func (l *LocalNodeClient) StatusOn(ctx context.Context, handle backend.Handle) (string, error) {
	return l.node.StatusOn(ctx, handle)
}

// RenewOwnerOn tells the node in this process that an owner is still running.
func (l *LocalNodeClient) RenewOwnerOn(_ context.Context, _ string, owner string) error {
	l.node.RenewOwner(owner)
	return nil
}

// ExecOn runs one command in a sandbox the node holds.
func (l *LocalNodeClient) ExecOn(
	ctx context.Context, handle backend.Handle, command, cwd string, env map[string]string, opts backend.ExecOptions,
) (backend.ExecResult, error) {
	return l.node.Exec(ctx, handle, command, cwd, env, opts)
}

// ReadBytesOn reads a file from a sandbox the node holds.
//
// The node owns the distinction between a missing file and a missing sandbox;
// returning its typed error lets the control plane preserve that classification
// across the JSON boundary instead of flattening both into "internal".
func (l *LocalNodeClient) ReadBytesOn(ctx context.Context, handle backend.Handle, path string) (string, error) {
	data, err := l.node.ReadBytes(ctx, handle, path)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

// WriteBytesOn writes a file into a sandbox the node holds.
func (l *LocalNodeClient) WriteBytesOn(ctx context.Context, handle backend.Handle, path, data string) error {
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return fmt.Errorf("writing %q: decode: %w", path, err)
	}
	return l.node.WriteBytes(ctx, handle, path, decoded)
}

// specToProto renders a spec for the node, which speaks the proto contract even
// in-process so the local and remote paths cannot diverge.
func specToProto(spec backend.Spec) *v1.SandboxSpec {
	out := &v1.SandboxSpec{
		Source:              &v1.Source{Kind: sourceKindValue(spec.Source.Kind), Reference: spec.Source.Reference},
		ResourceClass:       spec.ResourceClass,
		WorkflowId:          spec.WorkflowID,
		IdempotencyKey:      spec.IdempotencyKey,
		Env:                 spec.Env,
		Metadata:            spec.Metadata,
		Workdir:             spec.Workdir,
		ExecMode:            execModeValue(spec.ExecMode),
		RequiredResumeLevel: resumeLevelValue(spec.RequiredResume),
		Backend:             spec.Backend,
		RequiredNodeLabel:   spec.RequiredNodeLabel,
		ForbiddenNodeLabels: spec.ForbiddenNodeLabels,
		Resources:           &v1.Resources{},
	}
	if spec.Resources.CPUCount > 0 {
		cpu := spec.Resources.CPUCount
		out.Resources.CpuCount = &cpu
	}
	if spec.Resources.MemoryMB > 0 {
		memory := spec.Resources.MemoryMB
		out.Resources.MemoryMb = &memory
	}
	if spec.Resources.DiskMB > 0 {
		disk := spec.Resources.DiskMB
		out.Resources.DiskMb = &disk
	}
	if spec.Resources.GPUCount > 0 {
		gpus := spec.Resources.GPUCount
		out.Resources.GpuCount = &gpus
	}
	// The lifetime is how long the node will let the sandbox live, so dropping it
	// here would leave the node enforcing its own default instead of the one the
	// caller stated.
	if spec.Lifetime > 0 {
		out.LifetimeS = int64(spec.Lifetime / time.Second)
	}
	for _, feature := range spec.RequiredFeatures {
		out.RequiredFeatures = append(out.RequiredFeatures, featureValue(feature))
	}
	// Mounts travel the same way: a spec that asked for a host path and reached a node
	// without it would be a different environment, not the same one with less.
	for _, mount := range spec.Mounts {
		out.Mounts = append(out.Mounts, &v1.Mount{
			Source: mount.Source, Target: mount.Target, ReadOnly: mount.ReadOnly,
		})
	}
	if len(spec.BackendOptions) > 0 {
		out.BackendOptions = map[string]*v1.BackendOptions{}
		for name, values := range spec.BackendOptions {
			out.BackendOptions[name] = &v1.BackendOptions{Values: values}
		}
	}
	return out
}
