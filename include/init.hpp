#ifndef INIT_HPP
#define INIT_HPP

#include <SDL3/SDL.h>

namespace g
{
    constexpr int SCREEN_WIDTH {700};
    constexpr int SCREEN_HEIGHT {700};

}

bool init(SDL_Window*& w, SDL_Renderer*& r)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Could not initialize video! SDL error: %s\n", SDL_GetError());
        return false;
    }

    w = SDL_CreateWindow(
        "Digit Display",
        g::SCREEN_WIDTH,
        g::SCREEN_HEIGHT,
        0
    );

    if (w == nullptr)
    {
        SDL_Log("Could not initialize window! SDL error: %s\n", SDL_GetError());
        return false;
    }

    r = SDL_CreateRenderer(w, NULL);

    if (r == nullptr)
    {
        SDL_Log("Could not initialize SDL renderer! SDL error: %s\n", SDL_GetError());
        return false;
    }

    return true;
}

void close(SDL_Window*& w, SDL_Renderer*& r)
{
    SDL_DestroyRenderer(r);
    r = nullptr;

    SDL_DestroyWindowSurface(w);
    w = nullptr;

    SDL_Quit();
}

#endif