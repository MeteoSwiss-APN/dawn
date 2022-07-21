#include "to_json.hpp"


void write_metrics(const std::string& file_name, const std::string& stencil_name, VerificationMetrics& data)
{
    std::ofstream outfile;

    if (std::filesystem::exists(file_name)) {
        outfile.open(file_name, std::ios_base::app);
    } else {
        outfile.open(file_name, std::ios_base::out);
    };

    outfile << "Stencil: " << stencil_name << std::endl;
    outfile << "    " << "Iteration: " << data.iteration << std::endl;
    outfile << "    " << "Verified: "  << std::boolalpha << data.isValid << std::endl;
};