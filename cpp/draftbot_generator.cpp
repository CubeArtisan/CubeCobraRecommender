#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <blockingconcurrentqueue.h>
#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

constexpr std::size_t MAX_IN_PACK = 24;
constexpr std::size_t MAX_SEEN = 400;
constexpr std::size_t MAX_PICKED = 48;
constexpr std::size_t NUM_LAND_COMBS = 8;

struct PyPick {
    // We manipulate it so the first card is always the one chosen to simplify the model's loss calculation.
    static constexpr std::int32_t chosen_card = 0;

    std::array<std::int32_t, MAX_IN_PACK> in_pack{0};
    std::array<std::int32_t, MAX_SEEN> seen{0};
    float num_seen{0.f};
    std::array<std::int32_t, MAX_PICKED> picked{0};
    float num_picked{0.f};
    std::array<std::array<std::int32_t, 2>, 4> coords{{{0, 0}}};
    std::array<float, 4> coord_weights{0.f};
    std::array<std::array<std::uint8_t, MAX_SEEN>, NUM_LAND_COMBS> seen_probs{{{0}}};
    std::array<std::array<std::uint8_t, MAX_PICKED>, NUM_LAND_COMBS> picked_probs{{{0}}};
    std::array<std::array<std::uint8_t, MAX_IN_PACK>, NUM_LAND_COMBS> in_pack_probs{{{0}}};
    // std::array<std::array<float, MAX_SEEN>, NUM_LAND_COMBS> seen_probs{{{0.f}}};
    // std::array<std::array<float, MAX_PICKED>, NUM_LAND_COMBS> picked_probs{{{0.f}}};
    // std::array<std::array<float, MAX_IN_PACK>, NUM_LAND_COMBS> in_pack_probs{{{0.f}}};

    const char* load_from(const char* current_pos) {
        *this = {};
        for (std::size_t i = 0; i < 4; i++) {
            for (std::size_t j = 0; j < 2; j++) {
                coords[i][j] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        for (std::size_t i = 0; i < 4; i++) {
            coord_weights[i] = *reinterpret_cast<const float*>(current_pos);
            current_pos += sizeof(float);
        }
        std::uint16_t num_in_pack = *reinterpret_cast<const std::uint16_t*>(current_pos);
		current_pos += sizeof(std::uint16_t);
		num_picked = *reinterpret_cast<const std::uint16_t*>(current_pos);
		current_pos += sizeof(std::uint16_t);
		num_seen = *reinterpret_cast<const std::uint16_t*>(current_pos);
		current_pos += sizeof(std::uint16_t);
        for (std::int32_t i = 0; i < num_in_pack; i++) {
            in_pack[i] = *reinterpret_cast<const std::uint16_t*>(current_pos);
            current_pos += sizeof(std::uint16_t);
        }
        for (std::size_t i = 0; i < num_in_pack; i++) {
            for (std::size_t j = 0; j < NUM_LAND_COMBS; j++) {
                in_pack_probs[j][i] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        for (std::int32_t i = 0; i < num_picked; i++) {
            picked[i] = *reinterpret_cast<const std::uint16_t*>(current_pos);
            current_pos += sizeof(std::uint16_t);
        }
        for (std::size_t i = 0; i < num_picked; i++) {
            for (std::size_t j = 0; j < NUM_LAND_COMBS; j++) {
                picked_probs[j][i] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        for (std::int32_t i = 0; i < num_seen; i++) {
            seen[i] = *reinterpret_cast<const std::uint16_t*>(current_pos);
            current_pos += sizeof(std::uint16_t);
        }
        for (std::size_t i = 0; i < num_seen; i++) {
            for (std::size_t j = 0; j < NUM_LAND_COMBS; j++) {
                seen_probs[j][i] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        return current_pos;
    }
};

struct PyPickBatch {
    using python_type = std::tuple<py::array_t<std::int32_t>, py::array_t<std::int32_t>,
        py::array_t<float>, py::array_t<std::int32_t>, py::array_t<float>,
        py::array_t<std::int32_t>, py::array_t<float>, py::array_t<std::uint8_t>,
        py::array_t<std::uint8_t>, py::array_t<std::uint8_t>>;
    // using python_type = std::tuple<py::array_t<std::int32_t>, py::array_t<std::int32_t>,
    //     py::array_t<float>, py::array_t<std::int32_t>, py::array_t<float>,
    //     py::array_t<std::int32_t>, py::array_t<float>, py::array_t<float>,
    //     py::array_t<float>, py::array_t<float>>;

    constexpr std::array<std::size_t, 2> in_pack_shape() const noexcept { return { batch_size, MAX_IN_PACK }; }
    constexpr std::array<std::size_t, 2> seen_shape() const noexcept { return { batch_size, MAX_SEEN }; }
    constexpr std::array<std::size_t, 1> num_seen_shape() const noexcept { return { batch_size }; }
    constexpr std::array<std::size_t, 2> picked_shape() const noexcept { return { batch_size, MAX_PICKED }; }
    constexpr std::array<std::size_t, 1> num_picked_shape() const noexcept { return { batch_size }; }
    constexpr std::array<std::size_t, 3> coords_shape() const noexcept { return { batch_size, 4, 2 }; }
    constexpr std::array<std::size_t, 2> coord_weights_shape() const noexcept { return { batch_size, 4 }; }
    constexpr std::array<std::size_t, 3> seen_probs_shape() const noexcept { return { batch_size, NUM_LAND_COMBS, MAX_SEEN }; }
    constexpr std::array<std::size_t, 3> picked_probs_shape() const noexcept { return { batch_size, NUM_LAND_COMBS, MAX_PICKED }; }
    constexpr std::array<std::size_t, 3> in_pack_probs_shape() const noexcept { return { batch_size, NUM_LAND_COMBS, MAX_IN_PACK }; }

    static constexpr std::array<std::size_t, 2> in_pack_strides{ sizeof(PyPick), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 2> seen_strides{ sizeof(PyPick), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 1> num_seen_strides{ sizeof(PyPick) };
    static constexpr std::array<std::size_t, 2> picked_strides{ sizeof(PyPick), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 1> num_picked_strides{ sizeof(PyPick) };
    static constexpr std::array<std::size_t, 3> coords_strides{ sizeof(PyPick), sizeof(std::array<std::int32_t, 2>), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 2> coord_weights_strides{ sizeof(PyPick), sizeof(float) };
    // static constexpr std::array<std::size_t, 3> seen_probs_strides{ sizeof(PyPick), sizeof(std::array<float, MAX_SEEN>), sizeof(float) };
    // static constexpr std::array<std::size_t, 3> picked_probs_strides{ sizeof(PyPick), sizeof(std::array<float, MAX_PICKED>), sizeof(float) };
    // static constexpr std::array<std::size_t, 3> in_pack_probs_strides{ sizeof(PyPick), sizeof(std::array<float, MAX_IN_PACK>), sizeof(float) };
    static constexpr std::array<std::size_t, 3> seen_probs_strides{ sizeof(PyPick), sizeof(std::array<std::uint8_t, MAX_SEEN>), sizeof(std::uint8_t) };
    static constexpr std::array<std::size_t, 3> picked_probs_strides{ sizeof(PyPick), sizeof(std::array<std::uint8_t, MAX_PICKED>), sizeof(std::uint8_t) };
    static constexpr std::array<std::size_t, 3> in_pack_probs_strides{ sizeof(PyPick), sizeof(std::array<std::uint8_t, MAX_IN_PACK>), sizeof(std::uint8_t) };

    constexpr std::size_t size() const noexcept { return batch_size; }

    auto begin() noexcept { return storage.begin(); }
    auto begin() const noexcept { return storage.begin(); }
	auto end() noexcept { return storage.end(); }
    auto end() const noexcept { return storage.end(); }

    PyPickBatch(std::size_t picks_per_batch)
            : batch_size(picks_per_batch), storage(batch_size)
    { }

    PyPickBatch(PyPickBatch&& other)
            : batch_size(other.batch_size), storage(std::move(other.storage))
    { }

    PyPickBatch& operator=(PyPickBatch&& other) {
        batch_size = other.batch_size;
        storage = std::move(other.storage);
        return *this;
    }

    PyPick& operator[](std::size_t index) {
        return storage[index];
    }

    python_type to_numpy(py::capsule& capsule) noexcept {
        PyPick& pick = storage.front();
        return {
            py::array_t<std::int32_t>(in_pack_shape(), in_pack_strides, reinterpret_cast<std::int32_t*>(pick.in_pack.data()), capsule),
            py::array_t<std::int32_t>(seen_shape(), seen_strides, reinterpret_cast<std::int32_t*>(pick.seen.data()), capsule),
            py::array_t<float>(num_seen_shape(), num_seen_strides, reinterpret_cast<float*>(&pick.num_seen), capsule),
            py::array_t<std::int32_t>(picked_shape(), picked_strides, reinterpret_cast<std::int32_t*>(pick.picked.data()), capsule),
            py::array_t<float>(num_picked_shape(), num_picked_strides, reinterpret_cast<float*>(&pick.num_picked), capsule),
            py::array_t<std::int32_t>(coords_shape(), coords_strides, reinterpret_cast<std::int32_t*>(pick.coords.data()), capsule),
            py::array_t<float>(coord_weights_shape(), coord_weights_strides, reinterpret_cast<float*>(pick.coord_weights.data()), capsule),
            py::array_t<std::uint8_t>(seen_probs_shape(), seen_probs_strides, reinterpret_cast<std::uint8_t*>(pick.seen_probs.data()), capsule),
            py::array_t<std::uint8_t>(picked_probs_shape(), picked_probs_strides, reinterpret_cast<std::uint8_t*>(pick.picked_probs.data()), capsule),
            py::array_t<std::uint8_t>(in_pack_probs_shape(), in_pack_probs_strides, reinterpret_cast<std::uint8_t*>(pick.in_pack_probs.data()), capsule)
        };
    }

    python_type to_numpy(py::capsule& capsule) const noexcept {
        const PyPick& pick = storage.front();
        return {
            py::array_t<std::int32_t>(in_pack_shape(), in_pack_strides, reinterpret_cast<const std::int32_t*>(pick.in_pack.data()), capsule),
            py::array_t<std::int32_t>(seen_shape(), seen_strides, reinterpret_cast<const std::int32_t*>(pick.seen.data()), capsule),
            py::array_t<float>(num_seen_shape(), num_seen_strides, reinterpret_cast<const float*>(&pick.num_seen), capsule),
            py::array_t<std::int32_t>(picked_shape(), picked_strides, reinterpret_cast<const std::int32_t*>(pick.picked.data()), capsule),
            py::array_t<float>(num_picked_shape(), num_picked_strides, reinterpret_cast<const float*>(&pick.num_picked), capsule),
            py::array_t<std::int32_t>(coords_shape(), coords_strides, reinterpret_cast<const std::int32_t*>(pick.coords.data()), capsule),
            py::array_t<float>(coord_weights_shape(), coord_weights_strides, reinterpret_cast<const float*>(pick.coord_weights.data()), capsule),
            py::array_t<std::uint8_t>(seen_probs_shape(), seen_probs_strides, reinterpret_cast<const std::uint8_t*>(pick.seen_probs.data()), capsule),
            py::array_t<std::uint8_t>(picked_probs_shape(), picked_probs_strides, reinterpret_cast<const std::uint8_t*>(pick.picked_probs.data()), capsule),
            py::array_t<std::uint8_t>(in_pack_probs_shape(), in_pack_probs_strides, reinterpret_cast<const std::uint8_t*>(pick.in_pack_probs.data()), capsule)
        };
    }

private:
    std::size_t batch_size;
    std::vector<PyPick> storage;
};

struct DraftPickGenerator {
    using result_type = typename PyPickBatch::python_type;
    static constexpr std::size_t read_buffer_count = (1ull << 18) / sizeof(PyPick); // 256 KB
    static constexpr std::size_t shuffle_read_buffer_count = (1ull << 22) / sizeof(PyPick); // 4 MB
    static constexpr std::size_t buffered_pick_count = (1ull << 32) / sizeof(PyPick); // 4 GB
    static constexpr std::size_t shuffle_buffer_count = (1ull << 29) / sizeof(PyPick); // 512 MB

    DraftPickGenerator(std::size_t picks_per_batch, std::size_t num_readers, std::size_t num_shufflers,
                       std::size_t num_batchers, std::size_t seed, const std::string& folder_path)
            : batch_size(picks_per_batch), num_reader_threads{num_readers}, num_shuffler_threads{num_shufflers},
              num_batch_threads{num_batchers},
              initial_seed{seed}, length{0}, loaded_batches{buffered_pick_count / picks_per_batch},
              files_to_read_producer{files_to_read}, loaded_batches_consumer{loaded_batches},
              main_rng{initial_seed, num_readers + num_shufflers + num_batchers} {
        py::gil_scoped_release release;
        std::cout << "\tbuffered_pick_count: " << buffered_pick_count << " in batches: " << buffered_pick_count / batch_size
                  << "\n\tshuffle_buffer_count: " << shuffle_buffer_count
                  << "\n\tread_buffer_count: " << read_buffer_count
                  << "\n\tshuffle_read_buffer_count: " << shuffle_read_buffer_count << std::endl;
        std::vector<char> loaded_file_buffer;
        for (const auto& path_data : std::filesystem::directory_iterator(folder_path)) {
            draft_filenames.push_back(path_data.path().string());
            std::ifstream picks_file(path_data.path(), std::ios::binary | std::ios::ate);
            auto file_size = picks_file.tellg();
            loaded_file_buffer.clear();
            loaded_file_buffer.resize(file_size);
            picks_file.seekg(0);
            picks_file.read(loaded_file_buffer.data(), file_size);
            const char* current_pos = loaded_file_buffer.data();
            const char* const end_pos = loaded_file_buffer.data() + loaded_file_buffer.size();
            const std::size_t prev_length = length;
            while (current_pos < end_pos) {
                length++;
                current_pos = skip_record(current_pos, end_pos);
            }
        }
    }

    DraftPickGenerator& enter() {
        py::gil_scoped_release release;
        if (exit_threads) {
            exit_threads = false;
            queue_new_epoch();
            std::size_t thread_number = 0;
            for (std::size_t i=0; i < num_reader_threads; i++) {
                reader_threads.emplace_back([this, j=thread_number++](){ this->read_worker(pcg32(this->initial_seed, j)); });
            }
            for (std::size_t i=0; i < num_shuffler_threads; i++) {
                shuffler_threads.emplace_back([this, j=thread_number++](){ this->shuffle_worker(pcg32(this->initial_seed, j)); });
            }
            for (std::size_t i=0; i < num_batch_threads; i++) {
                batch_threads.emplace_back([this, j=thread_number++](){ this->batch_worker(pcg32(this->initial_seed, j)); });
            }
        }
        return *this;
    }

    bool exit(py::object, py::object, py::object) {
        exit_threads = true;
        for (auto& worker : reader_threads) worker.join();
        for (auto& worker : shuffler_threads) worker.join();
        for (auto& worker : batch_threads) worker.join();
        return false;
    }

    std::size_t size() const noexcept { return (length + batch_size - 1) / batch_size; }

    DraftPickGenerator& queue_new_epoch() {
        std::shuffle(std::begin(draft_filenames), std::end(draft_filenames), main_rng);
        files_to_read.enqueue_bulk(files_to_read_producer, std::begin(draft_filenames), draft_filenames.size());
        if (files_to_read.size_approx() > 0) request_needed_work();
        return *this;
    }

    result_type next() {
        std::unique_ptr<PyPickBatch> batched;
        request_needed_work();
        if (!loaded_batches.try_dequeue(loaded_batches_consumer, batched)) {
            py::gil_scoped_release gil_release;
            do {
                request_needed_work();
                std::cout << "\nloaded_batches: " << loaded_batches.size_approx()
                          << ", loaded_picks: " << loaded_picks.size_approx() << ", shuffled_picks: " << shuffled_picks.size_approx()
                          << ", requested_picks: " << requested_picks << ", files_to_read: " << files_to_read.size_approx()
                          << std::endl;
            } while (!loaded_batches.wait_dequeue_timed(loaded_batches_consumer, batched, 10'000));
        }
        PyPickBatch* batched_ptr = batched.release();
        py::capsule free_when_done(batched_ptr, [](void* ptr) { delete reinterpret_cast<PyPickBatch*>(ptr); });
        return batched_ptr->to_numpy(free_when_done);
    }

    result_type getitem(std::size_t) { return next(); }

private:
    void request_needed_work() {
        std::size_t loaded_approx = batch_size * loaded_batches.size_approx() + loaded_picks.size_approx()
                                     + shuffled_picks.size_approx();
        if (loaded_approx < buffered_pick_count) {
            requested_picks = (buffered_pick_count - loaded_approx + read_buffer_count - 1) / read_buffer_count;
        }
        if (files_to_read.size_approx() < num_reader_threads) queue_new_epoch();
    }

    static const char* skip_record(const char* current_pos, const char* end_pos) noexcept {
        current_pos += sizeof(std::array<std::array<std::uint8_t, 2>, 4>) + sizeof(std::array<float, 4>);
        if (current_pos + 3 * sizeof(std::uint16_t) > end_pos) return end_pos + 1;
        std::uint16_t num_in_pack = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        std::uint16_t num_picked = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        std::uint16_t num_seen = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        return current_pos + (sizeof(std::uint16_t) + sizeof(std::array<std::uint8_t, NUM_LAND_COMBS>))
                              * (num_in_pack + num_picked + num_seen);
    }

    void read_worker(pcg32 rng) {
        std::vector<PyPick> picks_buffer(read_buffer_count);
        std::uniform_int_distribution<std::size_t> sleep_for(10'000, 90'000);
        moodycamel::ConsumerToken files_to_read_consumer(files_to_read);
        moodycamel::ProducerToken loaded_picks_producer(loaded_picks);
        std::string cur_filename;
        std::vector<char> loaded_file_buffer;
        std::size_t current_index = 0;
        while (!exit_threads) {
            while (!exit_threads && !files_to_read.wait_dequeue_timed(files_to_read_consumer, cur_filename, 20 * sleep_for(rng))) { }
            if (!exit_threads) {
                {
                    std::ifstream picks_file(cur_filename, std::ios::binary | std::ios::ate);
                    auto file_size = picks_file.tellg();
                    loaded_file_buffer.clear();
                    loaded_file_buffer.resize(file_size);
                    picks_file.seekg(0);
                    picks_file.read(loaded_file_buffer.data(), file_size);
                }
                const char* current_pos = loaded_file_buffer.data();
                const char* end_pos = loaded_file_buffer.data() + loaded_file_buffer.size();
                while (skip_record(current_pos, end_pos) <= end_pos && !exit_threads) {
                    current_pos = picks_buffer[current_index++].load_from(current_pos);
                    if (current_index >= picks_buffer.size()) {
                        std::size_t expected = requested_picks;
                        while (!exit_threads && (expected == 0 || requested_picks.compare_exchange_weak(expected, expected - 1))) {
                            if (expected == 0) {
                                std::this_thread::sleep_for(std::chrono::microseconds(sleep_for(rng)));
                                expected = requested_picks;
                            }
                        }
                        if (exit_threads) return;
                        loaded_picks.enqueue_bulk(loaded_picks_producer, std::begin(picks_buffer), picks_buffer.size());
                        current_index = 0;
                    }
                }
            }
        }
    }

    void shuffle_worker(pcg32 rng) {
        std::uniform_int_distribution<std::size_t> sleep_for(20'000, 180'000);
        moodycamel::ConsumerToken loaded_picks_consumer(loaded_picks);
        moodycamel::ProducerToken shuffled_picks_producer(shuffled_picks);
        std::vector<PyPick> read_buffer;
        read_buffer.reserve(shuffle_read_buffer_count);
        std::vector<PyPick> shuffle_buffer;
        shuffle_buffer.reserve(shuffle_buffer_count);
        while (!exit_threads && !loaded_picks.wait_dequeue_bulk_timed(loaded_picks_consumer, std::back_inserter(shuffle_buffer),
                                                                      shuffle_buffer.capacity() - shuffle_buffer.size(), sleep_for(rng))) { }
        std::shuffle(std::begin(shuffle_buffer), std::end(shuffle_buffer), rng);
        std::uniform_int_distribution<std::size_t> index_selector(0, shuffle_buffer.size() - 1);
        while (!exit_threads) {
            loaded_picks.wait_dequeue_bulk_timed(loaded_picks_consumer, std::back_inserter(read_buffer),
                                                 read_buffer.capacity() - read_buffer.size(), sleep_for(rng));
            if (!exit_threads && read_buffer.size() > 0) {
                for (std::size_t i = 0; i < read_buffer.size(); i++) {
                    std::swap(read_buffer[i], shuffle_buffer[index_selector(rng)]);
                }
                shuffled_picks.enqueue_bulk(shuffled_picks_producer, std::begin(read_buffer), read_buffer.size());
                read_buffer.clear();
            }
        }
    }

    void batch_worker(pcg32 rng) {
        std::uniform_int_distribution<std::size_t> sleep_for(50'000, 350'000);
        moodycamel::ConsumerToken shuffled_picks_consumer(shuffled_picks);
        moodycamel::ProducerToken loaded_batches_producer(loaded_batches);
        while (!exit_threads) {
            std::unique_ptr<PyPickBatch> batch = std::make_unique<PyPickBatch>(batch_size);
            auto iter = batch->begin();
            const auto end_iter = batch->end();
            while (!exit_threads && iter != end_iter) {
                iter += shuffled_picks.wait_dequeue_bulk_timed(shuffled_picks_consumer, iter,
                                                               std::distance(iter, end_iter), sleep_for(rng));
            }
            if (!exit_threads) {
                loaded_batches.enqueue(loaded_batches_producer, std::move(batch));
            }
        }
    }

    std::size_t batch_size;
    std::size_t num_reader_threads;
    std::size_t num_shuffler_threads;
    std::size_t num_batch_threads;
    std::size_t initial_seed;

    std::vector<std::string> draft_filenames;
    std::size_t length;

    moodycamel::BlockingConcurrentQueue<std::string> files_to_read;
    moodycamel::BlockingConcurrentQueue<PyPick> loaded_picks;
    moodycamel::BlockingConcurrentQueue<PyPick> shuffled_picks;
    moodycamel::BlockingConcurrentQueue<std::unique_ptr<PyPickBatch>> loaded_batches;
    moodycamel::ProducerToken files_to_read_producer;
    moodycamel::ConsumerToken loaded_batches_consumer;

    std::atomic<std::size_t> requested_picks;
    std::atomic<bool> exit_threads{true};
    std::vector<std::thread> reader_threads;
    std::vector<std::thread> shuffler_threads;
    std::vector<std::thread> batch_threads;

    pcg32 main_rng;
};

PYBIND11_MODULE(draftbot_generator, m) {
    using namespace pybind11::literals;
    py::class_<DraftPickGenerator>(m, "DraftPickGenerator")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, const std::string&>())
        .def("__enter__", &DraftPickGenerator::enter)
        .def("__exit__", &DraftPickGenerator::exit)
        .def("__len__", &DraftPickGenerator::size)
        .def("__getitem__", &DraftPickGenerator::getitem)
        .def("__next__", &DraftPickGenerator::next)
        .def("__iter__", &DraftPickGenerator::queue_new_epoch)
        .def("on_epoch_end", &DraftPickGenerator::queue_new_epoch)
        .def("queue_new_epoch", &DraftPickGenerator::queue_new_epoch);
}
