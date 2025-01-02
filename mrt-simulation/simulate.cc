#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <set>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include "platform_load_time_gen.hpp"

using std::string;
using std::unordered_map;
using std::vector;
using adjacency_matrix = std::vector<std::vector<size_t>>;

struct Platform {
    std::queue<int> holding_area;
    int platform; // id of train on platform, -1 if none
    int link; // id of train on link, -1 if none
    int p_ticks_left; // ticks left for boarding
    int l_ticks_left; // ticks left for link
    int next_station;
    PlatformLoadTimeGen pltg;
    Platform(size_t popularity)
        : platform(-1),         
          link(-1),             
          p_ticks_left(0),      
          l_ticks_left(0),      
          pltg(popularity) 
    {}
};

int number_trains = 0;
Platform ***platforms;
std::unordered_map<char, Platform**> *stations;
std::vector<char> train_ids_to_line; // mapping of train_ids to line
std::vector<std::vector<int>> process_to_station_ids;
const vector<string> station_names;
const std::vector<size_t> popularities;
const adjacency_matrix mat;
const unordered_map<char, vector<string>> station_lines;

void send_trains(std::vector<std::vector<std::array<int, 2>>> arrival_queues, int recv_rank) {
    // Flatten the data into a 1D array
    std::vector<int> flattened;
    std::vector<int> row_sizes; // Store the size of each row for reconstruction
    for (const auto& row : arrival_queues) {
        row_sizes.push_back(row.size());
        for (const auto& train : row) {
            flattened.push_back(train[0]);
            flattened.push_back(train[1]);
        }
    }

    // Send the sizes of each row
    MPI_Bsend(row_sizes.data(), arrival_queues.size(), MPI_INT, recv_rank, 1, MPI_COMM_WORLD);

    // Send the flattened data
    MPI_Bsend(flattened.data(), flattened.size(), MPI_INT, recv_rank, 2, MPI_COMM_WORLD);
}

std::vector<std::vector<std::array<int, 2>>> recv_trains(int source_rank, int num_rows) {

    // Receive the sizes of each row
    std::vector<int> row_sizes(num_rows);
    MPI_Recv(row_sizes.data(), num_rows, MPI_INT, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Calculate the total number of elements to receive
    int total_elements = 0;
    for (int size : row_sizes) {
        total_elements += size * 2; // Each `std::array<int, 2>` contributes 2 integers
    }

    // Receive the flattened data
    std::vector<int> flattened(total_elements);
    MPI_Recv(flattened.data(), total_elements, MPI_INT, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Reconstruct the 2D vector
    std::vector<std::vector<std::array<int, 2>>> arrival_queues;
    int idx = 0;
    for (int row_size : row_sizes) {
        std::vector<std::array<int, 2>> row;
        for (int i = 0; i < row_size; ++i) {
            row.push_back({flattened[idx], flattened[idx + 1]});
            idx += 2;
        }
        arrival_queues.push_back(row);
    }

    return arrival_queues;
}

void generate_output(
    const vector<string> station_names, 
    int mpi_rank,
    size_t num_stations, 
    size_t current_tick, 
    int total_processes) 
{
        std::vector<int> train_positions;

        // Iterate over all platforms
        for (int station_id : process_to_station_ids[mpi_rank]) {
            for (size_t next_station_id = 0; next_station_id < num_stations; ++next_station_id) {
                Platform* platform = platforms[station_id][next_station_id];

                // Check if platform exists
                if (!platform) continue;

                // Check train on platform
                if (platform->platform != -1) {
                    int train_id = platform->platform;
                    train_positions.push_back(train_id);
                    train_positions.push_back(station_id);
                    train_positions.push_back(next_station_id);
                    train_positions.push_back(0); // 0 for platform
                }

                // Check train on link
                if (platform->link != -1) {
                    int train_id = platform->link;
                    train_positions.push_back(train_id);
                    train_positions.push_back(station_id);
                    train_positions.push_back(next_station_id);
                    train_positions.push_back(1); // 1 for link
                }

                // Check holding area
                std::queue<int> holding_queue = platform->holding_area;
                while (!holding_queue.empty()) {
                    int train_id = holding_queue.front();
                    holding_queue.pop();
                    train_positions.push_back(train_id);
                    train_positions.push_back(station_id);
                    train_positions.push_back(next_station_id);
                    train_positions.push_back(2); // 2 for holding area
                }
            }
        }

        int local_size = train_positions.size();

        int total_sizes[total_processes];
        int displ[total_processes];
        
        MPI_Gather(&local_size, 1, MPI_INT, total_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int sum = 0;

        if (mpi_rank == 0) {
            for (int i = 0; i < total_processes; i++) {
                displ[i] = sum;
                sum += total_sizes[i];
            }
        }

        int total_train_positions[number_trains * 4];

        MPI_Gatherv(train_positions.data(), local_size, MPI_INT, total_train_positions, total_sizes, displ, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<std::string> output;

        if (mpi_rank == 0) {
            for (int i = 0; i < number_trains * 4; i += 4) {
                int train_id = total_train_positions[i];
                int station_id = total_train_positions[i + 1];
                int next_station_id = total_train_positions[i + 2];
                int position = total_train_positions[i + 3];
                char line = train_ids_to_line[train_id];

                if (position == 0) { // 0 for platform
                    output.push_back(line + std::to_string(train_id) + "-" + station_names[station_id] + "%");
                } else if (position == 1) { // 1 for link
                    output.push_back(line + std::to_string(train_id) + "-" + station_names[station_id] + "->" + station_names[next_station_id]);
                } else { // 2 for holding area
                    output.push_back(line + std::to_string(train_id) + "-" + station_names[station_id] + "#");
                }
            }
            // Sort train positions lexicographically
            std::sort(output.begin(), output.end());

            // Print for the current tick
            std::cout << current_tick << ": ";
            for (size_t k = 0; k < output.size(); ++k) {
                std::cout << output[k];
                if (k != output.size() - 1) std::cout << " ";
            }
            std::cout << std::endl;
        }
}


void simulate(size_t num_stations, const vector<string> &station_names, const std::vector<size_t> &popularities,
              const adjacency_matrix &mat, const unordered_map<char, vector<string>> &station_lines, size_t ticks,
              const unordered_map<char, size_t> num_trains, size_t num_ticks_to_print, size_t mpi_rank,
              size_t total_processes) {
    // TODO: Implement this with MPI using total_processes
    process_to_station_ids.resize(total_processes);
    for (int station_id = 0; station_id < num_stations; ++station_id) {
        int process_id = station_id % total_processes; // Assign stations cyclically
        process_to_station_ids[process_id].push_back(station_id);
    }

    int num_stations_local = process_to_station_ids[mpi_rank].size();
    
    char lines[3] = {'g', 'y', 'b'};

    // instantiate all platforms
    platforms = (Platform ***) malloc(sizeof(Platform **) * num_stations);
    stations = new unordered_map<char, Platform**> [num_stations];

    for (int i = 0; i < num_stations; i++) {
        platforms[i] = (Platform **) malloc(sizeof(Platform *) * num_stations);
        for (int j = 0; j <= i; j++) {
            if (mat[i][j] > 0) {
                platforms[i][j] = new Platform(popularities[i]); // forward direction
                platforms[j][i] = new Platform(popularities[j]); // backward direction
            } else {
                platforms[i][j] = nullptr;
                platforms[j][i] = nullptr;
            }
        }
    }

    // Map each station name to its station ID for quick lookups
    std::unordered_map<std::string, int> station_to_id;
    for (int id = 0; id < num_stations; ++id) {
        station_to_id[station_names[id]] = id;
    }

    std::unordered_map<char, vector<int>> line_station_ids;

    for (const auto& line: station_lines) {
        const char line_name = line.first;
        const vector<string> station_names = line.second;
        std::vector<int> station_ids;

        for (string station_name: station_names) {
            station_ids.push_back(station_to_id[station_name]);
        }
        line_station_ids[line_name] = station_ids;
    }

    for (const auto& line: station_lines) {
        const char line_name = line.first;
        const vector<string> station_names = line.second; // vector of station names for that line
        int len = station_names.size();

        // iterate through all stations in line and add line details to platform
        for (int i = 0; i < len; i++) {
            int curr_station_id = station_to_id[station_names[i]];
            stations[curr_station_id][line_name] = (Platform **) malloc(sizeof(Platform *) * 2);
            if (i != len - 1) {
                int next_station_id = station_to_id[station_names[i + 1]];
                platforms[curr_station_id][next_station_id]->next_station = next_station_id; //forward

                stations[curr_station_id][line_name][0] = platforms[curr_station_id][next_station_id];
                 
            } else {
                stations[curr_station_id][line_name][0] = nullptr;
            }
            if (i != 0) {
                int prev_station_id = station_to_id[station_names[i - 1]];
                platforms[curr_station_id][prev_station_id]->next_station = prev_station_id; //backward

                stations[curr_station_id][line_name][1] = platforms[curr_station_id][prev_station_id];
            } else {
                stations[curr_station_id][line_name][1] = nullptr;
            }
        }
    }

    bool fully_spawned = false;
    unordered_map<char, size_t> num_trains_map(num_trains);

    for (size_t t = 0; t < ticks; t++) {
        MPI_Barrier(MPI_COMM_WORLD);
        // empty queues
        std::vector<std::vector<std::array<int, 2>>> arrival_queues(num_stations);

        // visited table so that platforms will not be visited twice
        std::vector<std::vector<bool>> visited1(num_stations, std::vector<bool>(num_stations, false));
        std::vector<std::vector<bool>> visited2(num_stations, std::vector<bool>(num_stations, false));


        // move trains
        for (const auto& station_id: process_to_station_ids[mpi_rank]) {
            for (auto &line_entry : stations[station_id]) {
                char line = line_entry.first;
                for (int i = 0; i < 2; i++) {
                    Platform *platform = line_entry.second[i];

                    if (!platform) continue;

                    if (visited1[station_id][platform->next_station]) continue;

                    // Update ticks left for outgoing link travel
                    if (platform->link != -1) {
                        if (train_ids_to_line[platform->link] != line) continue;
                        platform->l_ticks_left--;
                        if (platform->l_ticks_left == 0) {
                            int train_id = platform->link;
                            int next_station_id = platform->next_station;
                            int direction = i;

                            // check if terminal station
                            if (next_station_id == line_station_ids.at(train_ids_to_line[train_id])[0]) {
                                direction = 0;
                            } else if (next_station_id == line_station_ids.at(train_ids_to_line[train_id]).back()) {
                                direction = 1;
                            }
                            arrival_queues[next_station_id].push_back({train_id, direction}); // Send train to another station
                            platform->link = -1; // Link becomes free
                        }
                    }

                    // Update ticks left for loading/unloading
                    if (platform->platform != -1 && platform->p_ticks_left != 0) {
                        platform->p_ticks_left--;
                    }

                    visited1[station_id][platform->next_station] = true;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int recv_rank = 0; recv_rank < total_processes; recv_rank++) {
            if (recv_rank == mpi_rank) continue;
            std::vector<int> stations_to_send = process_to_station_ids[recv_rank];
            std::vector<std::vector<std::array<int, 2>>> send_buffer(stations_to_send.size());
            int idx = 0;
            for (int station_id: stations_to_send) {
                send_buffer[idx].insert(send_buffer[idx].end(), arrival_queues[station_id].begin(), arrival_queues[station_id].end());
                idx++;
            }
            send_trains(send_buffer, recv_rank);
        }


        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<std::vector<std::array<int, 2>>> recv_buffer(num_stations_local);
        for (int sender_rank = 0; sender_rank < total_processes; sender_rank++) {
            if (sender_rank == mpi_rank) continue;

            int num_rows = num_stations_local;
            
            std::vector<std::vector<std::array<int, 2>>> res = recv_trains(sender_rank, num_rows);
            // Append the received data to local arrival_queues
            for (size_t i = 0; i < num_stations_local; ++i) {
                int station_id = process_to_station_ids[mpi_rank][i];
                arrival_queues[station_id].insert(arrival_queues[station_id].end(), res[i].begin(), res[i].end());
            }
        }



        // spawn more trains if not fully spawned yet
        if (!fully_spawned) {
            fully_spawned = true;

            for (char line: lines) {
                int num_trains_to_initialize = std::min(num_trains_map[line], (size_t) 2);
                if (num_trains_to_initialize >= 1) {
                    std::string station_name = station_lines.at(line)[0];
                    int station_id = station_to_id[station_name];
                    int train_id = number_trains++;
                    arrival_queues[station_id].push_back({train_id, 0});
                    train_ids_to_line.push_back(line);
                }
                if (num_trains_to_initialize == 2) {
                    std::string station_name = station_lines.at(line)[station_lines.at(line).size() - 1];
                    int station_id = station_to_id[station_name];
                    int train_id = number_trains++;
                    arrival_queues[station_id].push_back({train_id, 1});
                    train_ids_to_line.push_back(line);
                }
                num_trains_map[line] -= num_trains_to_initialize;

                if (num_trains_map[line] > 0) {
                    fully_spawned = false;
                }
            }
        } 

        // sort arrival queues via train_id
        for (const auto& station_id: process_to_station_ids[mpi_rank]) {
            std::sort(arrival_queues[station_id].begin(), arrival_queues[station_id].end(), [](const std::array<int, 2>& a, const std::array<int, 2>& b) {
                return a[0] < b[0]; // Compare based on the first element
            });
        }

        // transfer sorted arrival queue to holding area
        for (const auto& station_id: process_to_station_ids[mpi_rank]) {
            for (const auto& train: arrival_queues[station_id]) {
                int train_id = train[0];
                int direction = train[1];

                stations[station_id][train_ids_to_line[train_id]][direction]->holding_area.push(train_id);
            }
        }

        // move trains from platform to link and holding area to platform
        for (const auto& station_id: process_to_station_ids[mpi_rank]) {
            for (auto &line_entry : stations[station_id]) {
                char line = line_entry.first;
                for (int direction = 0; direction < 2; direction++) {
                    Platform *platform = line_entry.second[direction];

                    if (!platform) continue;

                    

                    if (visited2[station_id][platform->next_station]) continue;

                    visited2[station_id][platform->next_station] = true;

                    // if link is empty and train on platform is ready to leave
                    if (platform->link == -1 && platform->platform != -1 && platform->p_ticks_left == 0) {
                        platform->link = platform->platform;
                        platform->platform = -1;
                        platform->l_ticks_left = mat[station_id][platform->next_station];
                    }

                    // if there is no train on platform and there is at least a train in holding area
                    if (platform->platform == -1 && !platform->holding_area.empty()) {
                        platform->platform = platform->holding_area.front();
                        platform->holding_area.pop();
                        platform->p_ticks_left = platform->pltg.next(platform->platform);
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (t >= ticks - num_ticks_to_print) {
            generate_output(station_names, mpi_rank, num_stations, t, total_processes);
        }

    }

    // Free the memory allocated for `platforms`
    for (int i = 0; i < num_stations; i++) {
        for (int j = 0; j < num_stations; j++) {
            // Free individual `Platform` objects
            if (platforms[i][j] != nullptr) {
                delete platforms[i][j];
            }
        }
        // Free the inner array of `Platform*`
        free(platforms[i]);
    }
    // Free the outer array of `Platform**`
    free(platforms);


    // Free the memory allocated for `stations`
    for (int i = 0; i < num_stations; i++) {
        for (auto& pair : stations[i]) {
            Platform** platform_ptrs = pair.second;

            // Free the array of `Platform*`
            if (platform_ptrs != nullptr) {
                free(platform_ptrs);
            }
        }
    }

    delete[] stations;

}