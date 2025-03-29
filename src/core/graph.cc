#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize0() {
        IT_ASSERT(topo_sort());

        for (auto it = ops.begin(); it != ops.end(); ) {
            auto op = *it;
            
            if (op->getOpType() == OpType::Transpose) {
                auto curTranspose = as<TransposeObj>(op);
                auto input = curTranspose->getInputs()[0];
                auto output = curTranspose->getOutput();
                
                if (auto producer = input->getSource()) {
                    if (producer->getOpType() == OpType::Transpose) {
                        auto preTranspose = std::dynamic_pointer_cast<TransposeObj>(producer);
                        
                        // Check if permutations are identical (would cancel out)
                        if (curTranspose->getPermute() == preTranspose->getPermute()) {
                            // Bypass both transposes
                            auto preInput = preTranspose->getInputs()[0];
                            preInput->removeTarget(preTranspose);
                            for (auto &target : curTranspose->getSuccessors()) {
                                target->removePredecessors(curTranspose);
                                target->replaceInput(output, preInput);
                                preInput->addTarget(target);
                            }
                            for (auto &source : preTranspose->getPredecessors()) {
                                source->removeSuccessors(preTranspose);
                            }
                            // Remove the transposes and intermediate tensor
                            removeTensor(input);
                            removeTensor(output);
                            removeOperator(preTranspose);
                            removeOperator(curTranspose);
                            continue;
                        }
                    }
                }
            } else if (op->getOpType() == OpType::MatMul) {
                auto matmul = as<MatmulObj>(op);
                auto inputs = matmul->getInputs();
                // bool modified = false;
                
                // Process both inputs (A and B)
                for (size_t i : {0, 1}) {
                    auto curInput = inputs[i];
                    // auto producer = op->getPredecessors()[i]; // <<== this leads to segmentation faults...
                    auto producer = curInput->getSource();
                    if (producer && producer->getOpType() == OpType::Transpose) {
                        auto transpose = as<TransposeObj>(producer);

                        // Check if transpose only swaps last two dims
                        bool canFuse = true;
                        auto perm = transpose->getPermute();
                        for (size_t j = 0; j < perm.size() - 2; j++) {
                            if (perm[j] != static_cast<int>(j)) {
                                canFuse = false;
                                break;
                            }
                        }
                        if (canFuse && perm.size() >= 2) {
                            canFuse = (perm[perm.size()-2] == static_cast<int>(perm.size()-1)) &&
                                    (perm[perm.size()-1] == static_cast<int>(perm.size()-2));
                        }
                        
                        if (canFuse) {
                            // Bypass the transpose
                            auto preInput = transpose->getInputs()[0];
                            preInput->removeTarget(transpose);

                            for (auto &source : transpose->getPredecessors()) {
                                source->removeSuccessors(transpose);
                            }
                            matmul->removePredecessors(transpose);
                            matmul->replaceInput(curInput, preInput);
                            
                            // Remove the transpose and intermediate tensor
                            removeOperator(transpose);
                            removeTensor(curInput);

                            // Toggle transpose flag
                            if(i) matmul->setTransB(!matmul->getTransB());
                            else matmul->setTransA(!matmul->getTransA());

                            --it;
                            continue;
                            // modified = true;
                        }
                    }
                }
                // if (modified) {
                //     it = ops.begin(); // Restart iteration if graph was modified
                //     continue;
                // }
            }
            ++it;
        }
    }

    void GraphObj::optimize1() {
        // https://github.com/haleihuixin/TinyInfiniTensor
        OpVec redundantOps;
        TensorVec redundantTensors;

        for (auto op : ops) {
            if (!op) continue;

            auto predecessors = op->getPredecessors();
            if (predecessors.size() == 1 && op->getOpType() == OpType::Transpose) {
                if (predecessors[0]->getOpType() == OpType::Transpose) {
                    auto curOp = std::dynamic_pointer_cast<TransposeObj>(op);
                    auto preOp = std::dynamic_pointer_cast<TransposeObj>(predecessors[0]);

                    if (curOp->getPermute() == preOp->getPermute()) {
                        auto preInput = preOp->getInputs()[0];
                        auto curInput = curOp->getInputs()[0];

                        preInput->removeTarget(preOp);
                        curOp->replaceInput(curInput, preInput);
                        
                        redundantTensors.push_back(curInput);
                        redundantOps.push_back(preOp);
                        redundantOps.push_back(curOp);
                    }
                }
            } else if (op->getOpType() == OpType::MatMul) {
                auto curOp = std::dynamic_pointer_cast<MatmulObj>(op);
                for (size_t i = 0; i < predecessors.size(); i++) {
                    bool flag = false;
                    if (predecessors[i]->getOpType() == OpType::Transpose) {
                        auto preOp = std::dynamic_pointer_cast<TransposeObj>(predecessors[i]);
                        auto permutation = preOp->getPermute();
                        if (permutation[permutation.size() - 1] == static_cast<int>(permutation.size() - 2) &&
                            permutation[permutation.size() - 2] == static_cast<int>(permutation.size() - 1)) {
                            auto it = std::find(redundantOps.begin(), redundantOps.end(), preOp);
                            if (it == redundantOps.end()) {
                                flag = true;
                                redundantOps.push_back(preOp);
                            }
                            
                            auto preInput = preOp->getInputs()[0];
                            auto curInput = curOp->getInputs()[i];
                            preInput->removeTarget(preOp);
                            preInput->addTarget(curOp);
                            curOp->replaceInput(curInput, preInput);
                            curOp->removePredecessors(preOp);
                            
                            redundantTensors.push_back(curInput);
                        }
                    }

                    if (flag) {
                        if (i) curOp->setTransB(!curOp->getTransB());
                        else curOp->setTransA(!curOp->getTransA());
                    }
                }
            }
        }

        for (auto op : redundantOps) {
            removeOperator(op);
        }
        for (auto tensor : redundantTensors) {
            removeTensor(tensor);
        }
    }

    void GraphObj::optimize2() {
        // https://github.com/Tunghohin/TinyInfiniTensor
        auto sources = this->getInputs();
        for (auto& tensor : sources) {
            WRef<TensorObj> current = refs_to_wrefs<TensorObj>({tensor})[0];
            while (current.lock()->getTargets().size()) {
                if (current.lock()->getTargets()[0]->getOpType() == OpType::Transpose && 
                    current.lock()->getTargets()[0]->getOutputs()[0]->getTargets().size()) {
                    WRef<OperatorObj> next_op = 
                        refs_to_wrefs<OperatorObj>({current.lock()->getTargets()[0]->getOutputs()[0]->getTargets()[0]})[0];
                    if (next_op.lock()->getOpType() == OpType::Transpose) {
                        auto current_matmulperm = ((TransposeObj*)current.lock()->getTargets()[0].get())->getPermute();
                        auto next_matmulperm = ((TransposeObj*)next_op.lock().get())->getPermute();
                        if (current_matmulperm == next_matmulperm) {
                            auto new_target =next_op.lock()->getOutputs()[0]->getTargets()[0]; 
                            new_target->removePredecessors(wrefs_to_refs<OperatorObj>({next_op})[0]);
                            current.lock()->addTarget(new_target);
                            new_target->replaceInput(next_op.lock()->getOutputs()[0], wrefs_to_refs<TensorObj>({current})[0]);

                            this->removeTensor(next_op.lock()->getOutputs()[0]);
                            this->removeTensor(current.lock()->getTargets()[0]->getOutputs()[0]);

                            auto removed_target = current.lock()->getTargets()[0];
                            current.lock()->removeTarget(removed_target);
                            this->removeOperator(wrefs_to_refs<OperatorObj>({next_op})[0]);
                            this->removeOperator(removed_target);
                        }
                    } else if (next_op.lock()->getOpType() == OpType::MatMul) {
                        auto current_matmulperm = ((TransposeObj*)current.lock()->getTargets()[0].get())->getPermute();
                        if ((size_t)current_matmulperm[current_matmulperm.size() - 1] == current_matmulperm.size() - 2 &&
                            (size_t)current_matmulperm[current_matmulperm.size() - 2] == current_matmulperm.size() - 1) {                            
                            auto removed_target = current.lock()->getTargets()[0];
                            auto new_target = removed_target->getOutputs()[0]->getTargets()[0];

                            new_target->replaceInput(removed_target->getOutputs()[0], wrefs_to_refs<TensorObj>({current})[0]);
                            if (new_target->inputs[0] == current.lock()) {
                                MatmulObj* new_target_ptr = (MatmulObj*)new_target.get();
                                new_target_ptr->setTransA(true);
                            } else {
                                MatmulObj* new_target_ptr = (MatmulObj*)new_target.get();
                                new_target_ptr->setTransB(true);
                            }
                            new_target->removePredecessors(removed_target);
                            current.lock()->removeTarget(removed_target);
                            current.lock()->addTarget(new_target);

                            this->removeTensor(removed_target->getOutputs()[0]);
                            this->removeOperator(removed_target);
                        }
                    }
                }
                current = current.lock()->getTargets()[0]->getOutputs()[0];
            }
        }
    }

    void GraphObj::optimize3() {
        // https://github.com/spacemaninhust/TinyInfiniTensor
        for (auto it = ops.begin(); it != ops.end(); )
        {
            auto op = *it;
            // 1. 去除冗余的算子
            if(!op) {
                it = ops.erase(it);
                continue;
            }
            if (op->getOpType() == OpType::Transpose)
            {
                auto input = op->getInputs()[0];
                auto output = op->getOutput();
                if (input->getTargets().size() == 1 &&
                    output->getTargets().size() == 1 &&
                    output->getTargets()[0]->getOpType() == OpType::Transpose)
                {
                    auto nextOp = output->getTargets()[0];
                    auto nextInput = nextOp->getInputs()[0];
                    auto nextOutput = nextOp->getOutput();
                    input->removeTarget(op);
                    nextInput->removeTarget(nextOp);
                    for (auto &pred: op->getPredecessors()) {
                        pred->removeSuccessors(op);
                    }
                    for (auto &succ: nextOp->getSuccessors()) {
                        succ->removePredecessors(nextOp);
                        succ->replaceInput(nextOutput, input);
                        input->addTarget(succ);
                    }
                    removeOperator(op);
                    removeOperator(nextOp);
                    removeTensor(output);
                    removeTensor(nextOutput);
                    continue;
                }
            } 
            // 2. 合并算子
            else if (op->getOpType() == OpType::MatMul) {
                auto matmul = as<MatmulObj>(op);
                auto inputs = matmul->getInputs();
                for(int i : {0, 1}) {
                    auto input = inputs[i];
                    if (input->getSource() && input->getSource()->getOpType() == OpType::Transpose) {
                        auto transpose = input->getSource();
                        auto transposeInput = transpose->getInputs()[0];
                        transposeInput->removeTarget(transpose);
                        for (auto &pred: transpose->getPredecessors()) {
                            pred->removeSuccessors(transpose);
                        }
                        matmul->removePredecessors(transpose);
                        removeOperator(transpose);
                        it -= 1;
                        matmul->replaceInput(input, transposeInput);
                        this->removeTensor(input);
                        if(i) matmul->setTransB(true);
                        else matmul->setTransA(true);
                        continue;
                    }
                }
            }
            it++;
        }
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        optimize0();
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        vector<size_t> offsets;
        for (const Tensor &tensor : tensors) {
            offsets.push_back(allocator.alloc(tensor->getBytes()));
        }

        void *ptr = allocator.getPtr();

        int size = offsets.size();
        for (int i = 0; i < size; ++i) {
            auto addr = reinterpret_cast<std::uintptr_t>(ptr) + offsets[i];
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, reinterpret_cast<void*>(addr)));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini