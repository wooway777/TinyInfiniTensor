#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // First-fit algorithm: find the first free block that's large enough
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            if (it->second >= size) {
                // Found a suitable block
                size_t addr = it->first;
                size_t remainingSize = it->second - size;
                
                // Remove the block from free list
                freeBlocks.erase(it);
                
                // If there's remaining space, add it back as a new free block
                if (remainingSize > 0) {
                    freeBlocks[addr + size] = remainingSize;
                }
                
                // Update memory usage statistics
                used += size;
                if (used > peak) {
                    peak = used;
                }
                
                return addr;
            }
        }
        
        // If no suitable free block found, allocate at the end
        size_t addr = peak;
        peak += size;
        used += size;

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        // Try to merge with adjacent free blocks
        auto next = freeBlocks.lower_bound(addr);
        auto prev = (next != freeBlocks.begin()) ? std::prev(next) : freeBlocks.end();

        // Check if we can merge with previous block
        if (prev != freeBlocks.end() && prev->first + prev->second == addr) {
            addr = prev->first;
            size += prev->second;
            freeBlocks.erase(prev);
        }

        // Check if we can merge with next block
        if (next != freeBlocks.end() && addr + size == next->first) {
            size += next->second;
            freeBlocks.erase(next);
        }

        // Add the merged block back to free list
        freeBlocks[addr] = size;

        // Update used memory
        used -= size;

        if (addr + size >= peak) {
            peak -= size;
            return;
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
