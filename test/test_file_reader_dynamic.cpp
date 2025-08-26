#include <iostream>
#include <fstream>
#include <regex>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <map>
#include <filesystem>

class TestDiscovery {
public:
    // Function to discover test functions by reading the source file
    static std::vector<std::string> discoverTestFunctions(const std::string& filename) {
        std::vector<std::string> testFunctions;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open file: " << filename << std::endl;
            return testFunctions;
        }
        
        std::string line;
        std::regex testRegex(R"(bool\s+(test_[a-zA-Z0-9_]+)\s*\()");
        std::smatch match;
        
        int lineNumber = 0;
        while (std::getline(file, line)) {
            lineNumber++;
            if (std::regex_search(line, match, testRegex)) {
                std::string functionName = match[1].str();
                testFunctions.push_back(functionName);
                std::cout << "🔍 Discovered test function: " << functionName 
                          << " (line " << lineNumber << ")" << std::endl;
            }
        }
        
        return testFunctions;
    }
};

class DynamicTestRunner {
private:
    std::map<std::string, void*> loadedLibraries;
    
public:
    ~DynamicTestRunner() {
        // Clean up loaded libraries
        for (auto& pair : loadedLibraries) {
            if (pair.second) {
                dlclose(pair.second);
            }
        }
    }
    
    // Find required source files based on the test file
    std::string findRequiredSourceFiles(const std::string& testFile) {
        std::vector<std::string> requiredSources;
        
        // Extract the test path components to map to source files
        // Example: "tensor/test_basic_tensor.cu" -> look for "../src/tensor/basic_tensor.cu"
        
        std::string testPath = testFile;
        
        // Remove "test_" prefix and ".cu" extension to get the base name
        size_t lastSlash = testPath.find_last_of('/');
        std::string fileName = (lastSlash != std::string::npos) ? testPath.substr(lastSlash + 1) : testPath;
        std::string directory = (lastSlash != std::string::npos) ? testPath.substr(0, lastSlash) : "";
        
        // Remove "test_" prefix if it exists
        if (fileName.length() > 5 && fileName.substr(0, 5) == "test_") {
            fileName = fileName.substr(5);
        }
        
        // Remove ".cu" extension
        if (fileName.length() > 3 && fileName.substr(fileName.length() - 3) == ".cu") {
            fileName = fileName.substr(0, fileName.length() - 3);
        }
        
        // Construct potential source file paths
        std::vector<std::string> candidatePaths;
        
        if (!directory.empty()) {
            // Same directory structure in src
            candidatePaths.push_back("../src/" + directory + "/" + fileName + ".cu");
            candidatePaths.push_back("../src/" + directory + "/" + fileName + ".cpp");
        }
        
        // Also try flat structure in src
        candidatePaths.push_back("../src/" + fileName + ".cu");
        candidatePaths.push_back("../src/" + fileName + ".cpp");
        
        // Check which files actually exist
        for (const std::string& candidate : candidatePaths) {
            if (fileExists(candidate)) {
                requiredSources.push_back(candidate);
                std::cout << "🔍 Found required source: " << candidate << std::endl;
            }
        }
        
        // Join all found sources with spaces
        std::string result;
        for (size_t i = 0; i < requiredSources.size(); ++i) {
            if (i > 0) result += " ";
            result += requiredSources[i];
        }
        
        return result;
    }
    
    // Helper function to check if a file exists
    bool fileExists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }
    
    // Compile a test file into a shared library
    bool compileToSharedLibrary(const std::string& sourceFile, const std::string& outputLib) {
        std::string compileCmd;
        
        // Check if it's a .cu file (CUDA) or .cpp file
        if (sourceFile.length() >= 3 && sourceFile.substr(sourceFile.length() - 3) == ".cu") {
            // Find potential source files that need to be included
            std::string additionalSources = findRequiredSourceFiles(sourceFile);
            
            if (!additionalSources.empty()) {
                compileCmd = "nvcc -shared -Xcompiler -fPIC -I../include -DSKIP_MAIN -DTEST_INSTANTIATIONS " + sourceFile + " " + additionalSources + " -o " + outputLib;
            } else {
                compileCmd = "nvcc -shared -Xcompiler -fPIC -I../include -DSKIP_MAIN -DTEST_INSTANTIATIONS " + sourceFile + " -o " + outputLib;
            }
        } else {
            compileCmd = "g++ -shared -fPIC -I../include " + sourceFile + " -o " + outputLib;
        }
        
        std::cout << "🔨 Compiling: " << compileCmd << std::endl;
        
        int result = system(compileCmd.c_str());
        if (result == 0) {
            std::cout << "✅ Successfully compiled " << sourceFile << " to " << outputLib << std::endl;
            return true;
        } else {
            std::cerr << "❌ Failed to compile " << sourceFile << std::endl;
            return false;
        }
    }
    
    // Load a shared library
    bool loadLibrary(const std::string& libPath) {
        void* handle = dlopen(libPath.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "❌ Cannot load library " << libPath << ": " << dlerror() << std::endl;
            return false;
        }
        
        loadedLibraries[libPath] = handle;
        std::cout << "✅ Successfully loaded library: " << libPath << std::endl;
        return true;
    }
    
    // Run a test function by name
    bool runTest(const std::string& functionName, const std::string& libPath) {
        auto it = loadedLibraries.find(libPath);
        if (it == loadedLibraries.end()) {
            std::cerr << "❌ Library not loaded: " << libPath << std::endl;
            return false;
        }
        
        void* handle = it->second;
        
        // Clear any existing error
        dlerror();
        
        // Get the function pointer
        typedef bool (*test_func_t)();
        test_func_t test_func = (test_func_t) dlsym(handle, functionName.c_str());
        
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "❌ Cannot load symbol " << functionName << ": " << dlsym_error << std::endl;
            return false;
        }
        
        // Call the function
        try {
            return test_func();
        } catch (const std::exception& e) {
            std::cerr << "❌ Exception in " << functionName << ": " << e.what() << std::endl;
            return false;
        }
    }
};

struct TestFile {
    std::string filename;
    std::string libraryPath;
    std::vector<std::string> test_functions;
    DynamicTestRunner* runner;

    TestFile(const std::string& filename, DynamicTestRunner* testRunner) 
        : filename(filename), runner(testRunner) {
        
        std::cout << "🔍 Discovering test functions in file: " << filename << std::endl;
        test_functions = TestDiscovery::discoverTestFunctions(filename);
        
        // Create shared_objects directory if it doesn't exist
        std::filesystem::create_directories("./shared_objects");
        
        // Generate library path in shared_objects folder
        std::filesystem::path filePath(filename);
        std::string baseName = filePath.stem().string();
        libraryPath = "./shared_objects/lib" + baseName + ".so";
        
        // Compile to shared library
        if (runner->compileToSharedLibrary(filename, libraryPath)) {
            runner->loadLibrary(libraryPath);
        }
    }

    bool runTest(const std::string& test_name) const {
        std::cout << "🏃 Attempting to run: " << test_name << " from " << libraryPath << std::endl;
        return runner->runTest(test_name, libraryPath);
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🔍 Dynamic Test Discovery and Execution" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "📁 Shared libraries will be stored in: ./shared_objects/" << std::endl;
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <test_file1> [test_file2] ..." << std::endl;
        return 1;
    }
    
    DynamicTestRunner runner;
    std::vector<TestFile> discoveredFiles;
    
    // Process each file argument
    for (int i = 1; i < argc; i++) {
        TestFile file(argv[i], &runner);
        discoveredFiles.push_back(file);
        std::cout << "📁 Analyzed: " << argv[i] << std::endl;
        std::cout << "  Found " << file.test_functions.size() << " test functions" << std::endl;
        std::cout << std::string(30, '-') << std::endl;
    }
    
    // Show discovery results
    std::cout << "📊 Discovery Results:" << std::endl;
    std::cout << "  Found " << discoveredFiles.size() << " files" << std::endl;
    for (const auto& file : discoveredFiles) {
        std::cout << "  " << file.filename << " - " << file.test_functions.size() << " test functions" << std::endl;
        for (const auto& test_name : file.test_functions) {
            std::cout << "    • " << test_name << std::endl;
        }
    }

    std::cout << std::endl;
    
    // Run all discovered test functions
    std::cout << "🧪 Running Tests:" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    int total = 0;
    int passed = 0;
    
    for (const auto& file : discoveredFiles) {
        for (const auto& test_name : file.test_functions) {
            total++;
            std::cout << "🏃 Running: " << file.filename << "::" << test_name << std::endl;
            
            // Actually call the test function using dynamic loading
            bool testResult = file.runTest(test_name);
            
            if (testResult) {
                passed++;
                std::cout << "✅ PASSED" << std::endl;
            } else {
                std::cout << "❌ FAILED" << std::endl;
            }
            std::cout << std::string(30, '-') << std::endl;
        }
    }
    
    // Summary
    std::cout << std::endl;
    std::cout << "📈 Final Results:" << std::endl;
    std::cout << "  Discovered: " << discoveredFiles.size() << " files" << std::endl;
    std::cout << "  Executed:   " << total << " tests" << std::endl;
    std::cout << "  Passed:     " << passed << " tests" << std::endl;
    std::cout << "  Success Rate: " << (total > 0 ? (passed * 100 / total) : 0) << "%" << std::endl;
    
    return (passed == total) ? 0 : 1;
}
