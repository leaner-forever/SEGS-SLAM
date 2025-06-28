#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>
#include <regex>
#include <string>
#include <algorithm>   // 用于 std::all_of

namespace example_utils
{
inline void LoadTrajectory(const std::string &filePath, std::vector<std::vector<double>> &data) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
}

inline void LoadTrajectory(const std::string &filePath, std::unordered_map<int, std::vector<double>> &data) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    int index = 0; // 行数索引
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data[index] = row; // 将行数据存储到哈希表中
            ++index; // 增加行数索引
        }
    }

    file.close();
}

inline void LoadTrajectory(const std::string &filePath, std::unordered_map<double, std::vector<double>> &data) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data[row[0]] = row; // 将行数据存储到哈希表中
            // std::cout << "LoadTrajectory: " << row[0] << std::endl;
        }
    }

    file.close();
}

inline void LoadTrajectoryInt(const std::string &filePath, std::unordered_map<int, std::vector<double>> &data) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data[int(row[0])] = row; // 将行数据存储到哈希表中
            // std::cout << "LoadTrajectory: " << row[0] << std::endl;
        }
    }

    file.close();
}

inline  int extractFrameNumber(const std::string& filename) {
    std::regex re("frame(\\d+)\\.jpg"); // 匹配 "frame" 后跟数字，再跟 ".jpg"
    std::smatch match;

    if (std::regex_search(filename, match, re)) {
        return std::stoi(match[1].str()); // 提取并转换为 int
    } else {
        throw std::invalid_argument("Filename format is incorrect");
    }
}


// 提取文件名中的数字部分（假设文件名格式为 "纯数字.jpg"）
inline int extract_number_from_filename(const std::string& filename) {
    // 获取不带扩展名的文件名（例如 "1.jpg" → "1"）
    std::filesystem::path p(filename);
    std::string stem = p.stem().string();

    // 验证文件名主干是否为纯数字
    if (stem.empty() || !std::all_of(stem.begin(), stem.end(), ::isdigit)) {
        throw std::invalid_argument("文件名格式错误: " + filename);
    }

    // 转换为整数
    return std::stoi(stem);
}
}