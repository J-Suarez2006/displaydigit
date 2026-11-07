#include "include/init.hpp"
#include "include/drawpixels.hpp"
#include "include/parsemnist.hpp"
#include "include/NN.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <chrono>

int main()
{
    constexpr int CELL_SIZE = 700 / 28;

    SDL_Window* window;
    SDL_Renderer* renderer;

    if(!init(window, renderer))
    {
        SDL_Log("Could not initialize! SDL Error: %s\n", SDL_GetError());
    }

    bool running {true};
    SDL_Event e;

    auto images = parseMnistImages("C:/Users/super/Dev/displaydigit/mnist_data/t10k-images.idx3-ubyte");

    digit_pixels pixels;
    for (auto& row : pixels)
    {
        row.fill(0);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, images.size() - 1);

    DigitNN NN;
    loadNN("C:\\Users\\super\\Dev\\displaydigit\\digit_nn.nn", NN);

    while (running)
    {
        while(SDL_PollEvent(&e))
        {
            if (e.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
            else if (e.type == SDL_EVENT_KEY_DOWN)
            {
                if (e.key.repeat == 0)
                {
                    if (e.key.key == SDLK_R)
                    {
                        int randomNumber = distribution(generator);
                        pixels = images[randomNumber];
                    }
                    if (e.key.key == SDLK_T)
                    {
                        Vec<10> output = NN.forward(getInputFromImage(pixels));
                        std::cout << "Detected a " << getOutput(output) << std::endl;
                    }
                }
            }
        }   
        
        float mouseX, mouseY;
        uint32_t buttons = SDL_GetMouseState(&mouseX, &mouseY);

        if (buttons & SDL_BUTTON_LEFT)
        {
            int cellx = (int)mouseX / CELL_SIZE;
            int celly = (int)mouseY / CELL_SIZE;

            pixels[celly][cellx] = uint8_t(255);
        }

        const bool* keystates = SDL_GetKeyboardState(nullptr);

        if (keystates[SDL_SCANCODE_SPACE])
        {
            int cellx = (int)mouseX / CELL_SIZE;
            int celly = (int)mouseY / CELL_SIZE;

            pixels[celly][cellx] = uint8_t(0); 
        }

        if (keystates[SDL_SCANCODE_LSHIFT])
        {
            for (auto& row : pixels)
            {
                row.fill(0);
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        drawPixels(pixels, window, renderer);
        SDL_RenderPresent(renderer);
    }

    close(window, renderer);

}