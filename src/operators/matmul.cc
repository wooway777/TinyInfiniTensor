#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // 1. valid input amount
        IT_ASSERT(inputs.size() == 2);
        const auto &A = inputs[0];
        const auto &B = inputs[1];
        
        // 2. retrieve input dimensions
        Shape shapeA = A->getDims();
        Shape shapeB = B->getDims();
        int rankA = shapeA.size();
        int rankB = shapeB.size();
        
        // 3. valid input dimensions
        IT_ASSERT(rankA >= 2 && rankB >= 2);
        
        // 4. handling broadcasting
        Shape batchDims;
        if (rankA > 2 || rankB > 2) {
            // get the size of the larger dimensions
            int maxBatchRank = std::max(rankA - 2, rankB - 2);
            for (int i = 0; i < maxBatchRank; ++i) {
                int dimA = (i < rankA - 2) ? shapeA[i] : 1;
                int dimB = (i < rankB - 2) ? shapeB[i] : 1;
                
                // check batch dimension compatibility
                IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1);
                batchDims.push_back(std::max(dimA, dimB));
            }
        }
        
        // 5. calculate multiplication dimensions
        int m = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        int kA = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
        int kB = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        int n = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];
        
        // 6. validate K dimensions
        IT_ASSERT(kA == kB);
        
        // 7. save information for further operations
        this->m = m;
        this->n = n;
        this->k = kA;
        
        // 8. construct the output shape
        Shape outputShape = batchDims;
        outputShape.push_back(m);
        outputShape.push_back(n);
        
        return {{outputShape}};
    }

} // namespace infini