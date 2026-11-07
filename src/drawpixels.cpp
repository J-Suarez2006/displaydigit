#include "../include/drawpixels.hpp"

void drawPixels(const digit_pixels& pixels, SDL_Window*& w, SDL_Renderer*& r)
{
    int screen_width, screen_height;
    SDL_GetWindowSize(w, &screen_width, &screen_height);

    int row_num {0};
    for (const auto& row : pixels)
    {
        int pixel_num {0};

        for (const auto& pixel : row)
        {
            SDL_FRect pixel[1]
            {
                {screen_width / 28 * pixel_num, screen_height / 28 * row_num, 100, 100}
            };

            int intensity = pixels[row_num][pixel_num];
            SDL_SetRenderDrawColor(r, intensity, intensity, intensity, 255);
            SDL_RenderFillRect(r, pixel);

            ++pixel_num;
        }
        
        ++row_num;
    }
}