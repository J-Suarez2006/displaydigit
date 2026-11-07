#ifndef DRAW_DIGIT
#define DRAW_DIGIT

#include <SDL3/SDL.h>
#include <array>


// should be 28 x 28
using digit_pixels = std::array<std::array<uint8_t, 28>, 28>;

void drawPixels(const digit_pixels& pixels, SDL_Window*& w, SDL_Renderer*& r);

#endif