#pragma once

#include <string>
#include <mutex>
#include "hmgs_config.h"

namespace spacego {
namespace algorithms {

class ConfigManager {
public:
    ConfigManager();
    explicit ConfigManager(const HMGSConfig& config);
    
    void set_config(const HMGSConfig& config);
    HMGSConfig get_config() const;
    
    void update_parameter(const std::string& param_name, double value);
    void update_parameter(const std::string& param_name, size_t value);
    void update_parameter(const std::string& param_name, const std::string& value);
    void update_parameter(const std::string& param_name, bool value);

private:
    HMGSConfig config_;
    mutable std::mutex mutex_;
};

} // namespace algorithms
} // namespace spacego
