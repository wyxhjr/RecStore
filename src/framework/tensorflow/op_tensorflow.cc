#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "framework/op.h"
#include "base/tensor.h"

namespace tf = tensorflow;

REGISTER_OP("RecstoreEmbRead")
    .Input("keys: uint64")
    .Attr("embedding_dim: int >= 1")
    .Output("values: float")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        tf::shape_inference::ShapeHandle keys_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &keys_shape));
        
        int64_t emb_dim_attr = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("embedding_dim", &emb_dim_attr));

        tf::shape_inference::DimensionHandle L = c->Dim(keys_shape, 0);
        tf::shape_inference::ShapeHandle values_shape = c->Matrix(L, emb_dim_attr);
        c->set_output(0, values_shape);
        return tf::OkStatus();
    })
    .Doc(R"doc(
Reads embedding vectors from Recstore.
keys: A uint64 ID tensor of length L.
embedding_dim: The embedding dimension D to read.
values: An embedding tensor with shape [L, D].
)doc");

REGISTER_OP("RecstoreEmbUpdate")
    .Input("keys: uint64")
    .Input("grads: float")
    .Attr("embedding_dim: int >= 1")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        return tf::OkStatus();
    })
    .Doc(R"doc(
Updates embedding vectors in Recstore.
keys: A uint64 ID tensor of length L.
embedding_dim: The embedding dimension D to update.
grads: A gradient tensor with shape [L, D].
)doc");

REGISTER_OP("RecstoreEmbWrite")
    .Input("keys: uint64")
    .Input("values: float")
    .Attr("embedding_dim: int >= 1")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        return tf::OkStatus();
    })
    .Doc(R"doc(
Writes embedding vectors to Recstore.
keys: A uint64 ID tensor of length L.
embedding_dim: The embedding dimension D to write.
values: An embedding tensor with shape [L, D].
)doc");

REGISTER_OP("RecstoreEmbPrefetch")
    .Input("keys: uint64")
    .Input("values: float")
    .Output("prefetch_id: uint64")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        tf::shape_inference::ShapeHandle keys_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &keys_shape));
        
        tf::shape_inference::DimensionHandle L = c->Dim(keys_shape, 0);
        tf::shape_inference::ShapeHandle prefetch_id_shape = c->Matrix(L, 1);
        c->set_output(0, prefetch_id_shape);
        return tf::OkStatus();
    })
    .Doc(R"doc(
Prefetches embedding vectors asynchronously from Recstore.
keys: A uint64 ID tensor of length L.
values: An embedding tensor with shape [L, D].
prefetch_id: A unique ID to track the prefetch status.
)doc");

REGISTER_OP("RecstoreIsPrefetchDone")
    .Input("prefetch_id: uint64")
    .Output("done: bool")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        tf::shape_inference::ShapeHandle prefetch_id_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &prefetch_id_shape));

        tf::shape_inference::ShapeHandle done_shape = c->Matrix(1, 1);
        c->set_output(0, done_shape);
        return tf::OkStatus();
    })
    .Doc(R"doc(
Checks if the prefetch operation identified by prefetch_id is done.
prefetch_id: A unique ID returned by EmbPrefetch.
done: A bool tensor indicating whether the prefetch operation is complete.
)doc");

REGISTER_OP("RecstoreWaitForPrefetch")
    .Input("prefetch_id: uint64")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        return tf::OkStatus();
    })
    .Doc(R"doc(
Blocks until the prefetch operation identified by prefetch_id is complete.
prefetch_id: A unique ID returned by EmbPrefetch.
)doc");

class RecstoreEmbReadOp : public tf::OpKernel {
public:
    explicit RecstoreEmbReadOp(tf::OpKernelConstruction* context) : OpKernel(context) {
        // Read embedding_dim attribute
        OP_REQUIRES_OK(context, context->GetAttr("embedding_dim", &emb_dim_));
        OP_REQUIRES(context, emb_dim_ > 0, tf::errors::InvalidArgument("embedding_dim must be > 0"));
    }

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& keys_tensor = context->input(0);
        OP_REQUIRES(context, tf::TensorShapeUtils::IsVector(keys_tensor.shape()),
                    tf::errors::InvalidArgument("Keys must be a 1-D vector."));

        const int64_t L = keys_tensor.dim_size(0);
        const int64_t D = static_cast<int64_t>(emb_dim_);

        tf::Tensor* values_tensor = nullptr;
        tf::TensorShape values_shape({L, D});
        OP_REQUIRES_OK(context, context->allocate_output(0, values_shape, &values_tensor));

        base::RecTensor rec_keys(
            (void*)keys_tensor.flat<tensorflow::uint64>().data(),
            {L},
            base::DataType::UINT64);

        base::RecTensor rec_values(
            (void*)values_tensor->flat<float>().data(),
            {L, D},
            base::DataType::FLOAT32);

        try {
            auto op = recstore::GetKVClientOp();
            op->EmbRead(rec_keys, rec_values);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore EmbRead failed: ", e.what()));
        }
    }
private:
    int64_t emb_dim_ = 0;
};


class RecstoreEmbUpdateOp : public tf::OpKernel {
public:
    explicit RecstoreEmbUpdateOp(tf::OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("embedding_dim", &emb_dim_));
        OP_REQUIRES(context, emb_dim_ > 0, tf::errors::InvalidArgument("embedding_dim must be > 0"));
    }

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& keys_tensor = context->input(0);
        const tf::Tensor& grads_tensor = context->input(1);

        OP_REQUIRES(context, tf::TensorShapeUtils::IsVector(keys_tensor.shape()),
                    tf::errors::InvalidArgument("Keys must be a 1-D vector."));
        OP_REQUIRES(context, tf::TensorShapeUtils::IsMatrix(grads_tensor.shape()),
                    tf::errors::InvalidArgument("Grads must be a 2-D matrix."));
        OP_REQUIRES(context, keys_tensor.dim_size(0) == grads_tensor.dim_size(0),
                    tf::errors::InvalidArgument("Keys and Grads must have the same size in dimension 0."));
        OP_REQUIRES(context, grads_tensor.dim_size(1) == emb_dim_,
                    tf::errors::InvalidArgument("Grads has wrong embedding dimension."));

        const int64_t L = keys_tensor.dim_size(0);
        const int64_t D = grads_tensor.dim_size(1);
        
        base::RecTensor rec_keys(
            (void*)keys_tensor.flat<tensorflow::uint64>().data(),
            {L},
            base::DataType::UINT64);

        base::RecTensor rec_grads(
            (void*)grads_tensor.flat<float>().data(),
            {L, D},
            base::DataType::FLOAT32);
        
        try {
            auto op = recstore::GetKVClientOp();
            op->EmbUpdate(rec_keys, rec_grads);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore EmbUpdate failed: ", e.what()));
        }
    }
private:
    int64_t emb_dim_ = 0;
};


class RecstoreEmbWriteOp : public tf::OpKernel {
public:
    explicit RecstoreEmbWriteOp(tf::OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("embedding_dim", &emb_dim_));
        OP_REQUIRES(context, emb_dim_ > 0, tf::errors::InvalidArgument("embedding_dim must be > 0"));
    }

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& keys_tensor = context->input(0);
        const tf::Tensor& values_tensor = context->input(1);

        OP_REQUIRES(context, tf::TensorShapeUtils::IsVector(keys_tensor.shape()),
                    tf::errors::InvalidArgument("Keys must be a 1-D vector."));
        OP_REQUIRES(context, tf::TensorShapeUtils::IsMatrix(values_tensor.shape()),
                    tf::errors::InvalidArgument("Values must be a 2-D matrix."));
        OP_REQUIRES(context, keys_tensor.dim_size(0) == values_tensor.dim_size(0),
                    tf::errors::InvalidArgument("Keys and Values must have the same size in dimension 0."));
        OP_REQUIRES(context, values_tensor.dim_size(1) == emb_dim_,
                    tf::errors::InvalidArgument("Values has wrong embedding dimension."));

        const int64_t L = keys_tensor.dim_size(0);
        const int64_t D = values_tensor.dim_size(1);
        
        base::RecTensor rec_keys(
            (void*)keys_tensor.flat<tensorflow::uint64>().data(),
            {L},
            base::DataType::UINT64);

        base::RecTensor rec_values(
            (void*)values_tensor.flat<float>().data(),
            {L, D},
            base::DataType::FLOAT32);
        
        try {
            auto op = recstore::GetKVClientOp();
            op->EmbWrite(rec_keys, rec_values);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore EmbWrite failed: ", e.what()));
        }
    }
private:
    int64_t emb_dim_ = 0;
};


// Initial version: RecstoreEmbPrefetchOp：need to consider whether to call multiple prefetches concurrently
class RecstoreEmbPrefetchOp : public tf::OpKernel {
public:
    explicit RecstoreEmbPrefetchOp(tf::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tf::OpKernelContext* context) override {
        // Not implemented yet
        context->SetStatus(tf::errors::Unimplemented("RecstoreEmbPrefetch is not implemented yet."));
    }
};


class RecstoreIsPrefetchDoneOp : public tf::OpKernel {
public:
    explicit RecstoreIsPrefetchDoneOp(tf::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& prefetch_id_tensor = context->input(0);
        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(prefetch_id_tensor.shape()),
                    tf::errors::InvalidArgument("Prefetch ID must be a scalar."));

        uint64_t prefetch_id = prefetch_id_tensor.scalar<uint64_t>()();

        tf::Tensor* done_tensor = nullptr;
        tf::TensorShape done_shape({1, 1});
        OP_REQUIRES_OK(context, context->allocate_output(0, done_shape, &done_tensor));

        bool done = false;
        try {
            done = recstore::IsPrefetchDone(prefetch_id);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore IsPrefetchDone failed: ", e.what()));
            return;
        }

        done_tensor->flat<bool>()(0) = done;
    }
};


class RecstoreWaitForPrefetchOp : public tf::OpKernel {
public:
    explicit RecstoreWaitForPrefetchOp(tf::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& prefetch_id_tensor = context->input(0);
        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(prefetch_id_tensor.shape()),
                    tf::errors::InvalidArgument("Prefetch ID must be a scalar."));

        uint64_t prefetch_id = prefetch_id_tensor.scalar<uint64_t>()();

        try {
            recstore::WaitForPrefetch(prefetch_id);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore WaitForPrefetch failed: ", e.what()));
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("RecstoreEmbRead").Device(tf::DEVICE_CPU), RecstoreEmbReadOp);
REGISTER_KERNEL_BUILDER(Name("RecstoreEmbUpdate").Device(tf::DEVICE_CPU), RecstoreEmbUpdateOp);
REGISTER_KERNEL_BUILDER(Name("RecstoreEmbWrite").Device(tf::DEVICE_CPU), RecstoreEmbWriteOp);
REGISTER_KERNEL_BUILDER(Name("RecstoreEmbPrefetch").Device(tf::DEVICE_CPU), RecstoreEmbPrefetchOp);
REGISTER_KERNEL_BUILDER(Name("RecstoreIsPrefetchDone").Device(tf::DEVICE_CPU), RecstoreIsPrefetchDoneOp);
REGISTER_KERNEL_BUILDER(Name("RecstoreWaitForPrefetch").Device(tf::DEVICE_CPU), RecstoreWaitForPrefetchOp);