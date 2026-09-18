struct VdamCandidateBlockTraceHeader
{
    char magic[16];
    std::uint32_t schema_version;
    std::uint32_t header_size;
    std::uint32_t record_size;
    std::uint32_t iteration;
    std::uint64_t record_count;
    std::uint64_t capacity;
    std::uint64_t reserved0;
    std::uint64_t reserved1;
};

struct VdamCandidateBlockTraceRecord
{
    std::uint64_t launch_sequence;
    std::int64_t particle_id;
    std::uint64_t block_start_globaltimer;
    std::uint64_t first_atomic_globaltimer;
    std::uint64_t block_end_globaltimer;
    std::uint32_t orientation_row;
    std::int32_t worker_id;
    std::int32_t class_id;
    std::uint32_t sm_id;
    std::uint32_t image_count;
    std::uint32_t iteration;
    std::uint32_t flags;
    std::uint32_t reserved;
};

static_assert(sizeof(VdamCandidateBlockTraceHeader) == 64,
              "candidate VDAM block trace header must be 64 bytes");
static_assert(sizeof(VdamCandidateBlockTraceRecord) == 72,
              "candidate VDAM block trace record must be 72 bytes");

class VdamCandidateBlockTraceWriter
{
public:
    bool requested()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        return requested_;
    }

    bool healthy()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        return !requested_ || healthy_;
    }

    std::uint32_t iteration()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        return header_.iteration;
    }

    bool reserve(std::uint64_t record_count, std::uint64_t* launch_sequence)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        if (!requested_ || !healthy_ || record_count == 0 ||
            reserved_records_ > header_.capacity ||
            record_count > header_.capacity - reserved_records_)
            return false;
        *launch_sequence = next_launch_++;
        reserved_records_ += record_count;
        return true;
    }

    bool append(const VdamCandidateBlockTraceRecord* records, std::uint64_t record_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!requested_ || !healthy_ || records == nullptr || record_count == 0)
            return false;
        output_.seekp(0, std::ios::end);
        output_.write(
            reinterpret_cast<const char*>(records),
            static_cast<std::streamsize>(record_count * sizeof(*records)));
        written_records_ += record_count;
        header_.record_count = written_records_;
        output_.seekp(0, std::ios::beg);
        output_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
        output_.flush();
        healthy_ = static_cast<bool>(output_);
        return healthy_;
    }

private:
    void initialize_locked()
    {
        if (initialized_) return;
        initialized_ = true;
        const char* path = std::getenv("RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE");
        if (path == nullptr || path[0] == '\0') return;
        requested_ = true;
        const char* iteration_text =
            std::getenv("RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_ITER");
        const char* capacity_text =
            std::getenv("RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPACITY");
        if (iteration_text == nullptr || iteration_text[0] == '\0') return;
        errno = 0;
        char* iteration_end = nullptr;
        const unsigned long iteration =
            std::strtoul(iteration_text, &iteration_end, 10);
        if (errno != 0 || iteration_end == iteration_text || *iteration_end != '\0' ||
            iteration == 0 || iteration > std::numeric_limits<std::uint32_t>::max())
            return;
        std::uint64_t capacity = 1000000;
        if (capacity_text != nullptr && capacity_text[0] != '\0')
        {
            errno = 0;
            char* capacity_end = nullptr;
            const unsigned long long parsed =
                std::strtoull(capacity_text, &capacity_end, 10);
            if (errno != 0 || capacity_end == capacity_text || *capacity_end != '\0' ||
                parsed == 0)
                return;
            capacity = static_cast<std::uint64_t>(parsed);
        }
        const char magic[16] = {
            'R', 'E', 'L', 'I', 'O', 'N', '_', 'V', 'D', 'A', 'M', '_', 'B', 'T', '1', '\0'};
        std::memcpy(header_.magic, magic, sizeof(magic));
        header_.schema_version = 1;
        header_.header_size = sizeof(header_);
        header_.record_size = sizeof(VdamCandidateBlockTraceRecord);
        header_.iteration = static_cast<std::uint32_t>(iteration);
        header_.record_count = 0;
        header_.capacity = capacity;
        header_.reserved0 = 0;
        header_.reserved1 = 0;
        output_.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!output_) return;
        output_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
        output_.flush();
        healthy_ = static_cast<bool>(output_);
    }

    std::mutex mutex_;
    bool initialized_ = false;
    bool requested_ = false;
    bool healthy_ = false;
    std::fstream output_;
    VdamCandidateBlockTraceHeader header_ = {};
    std::uint64_t next_launch_ = 0;
    std::uint64_t reserved_records_ = 0;
    std::uint64_t written_records_ = 0;
};

static VdamCandidateBlockTraceWriter& vdam_candidate_block_trace_writer()
{
    static VdamCandidateBlockTraceWriter writer;
    return writer;
}

static __device__ __forceinline__ std::uint64_t vdam_candidate_globaltimer()
{
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ __forceinline__ std::uint32_t vdam_candidate_smid()
{
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}
