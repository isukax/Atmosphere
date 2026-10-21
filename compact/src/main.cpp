#include "app/App.h"
#include <iostream>
#include <string>
int main(int argc, char **argv) {
    bool debug = false, hidden = false, smoke = false;
    uint32_t frames = 0;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--debug")
                debug = true;
            else if (a == "--hidden")
                hidden = true;
            else if (a == "--smoke") {
                smoke = true;
                hidden = true;
                frames = 8;
            } else if (a == "--frames" && i + 1 < argc)
                frames = static_cast<uint32_t>(std::stoul(argv[++i]));
            else
                throw std::invalid_argument("Unknown argument: " + a);
        }
        compact::app::App app(debug, hidden);
        return app.run(frames, smoke);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
