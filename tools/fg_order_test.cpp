// Minimal 30/60 FPS monotonic frame-order & latency test app for Gamescope FrameGen.
// Build: g++ -O2 tools/fg_order_test.cpp $(sdl2-config --cflags --libs) -o fg_order_test
// Run:   gamescope -w 1280 -h 720 -W 1920 -H 1200 -r 60 --framerate-limit 30 --frame-gen opticalflow -f -- ./fg_order_test
// Film with a 240 fps phone camera: the bar should move smoothly forward EVERY frame (never jumping backward).

#include <SDL.h>
#include <cstdio>
#include <cstdint>

static const uint8_t kDigitFont[10][5] = {
    {0x7, 0x5, 0x5, 0x5, 0x7}, // 0
    {0x2, 0x6, 0x2, 0x2, 0x7}, // 1
    {0x7, 0x1, 0x7, 0x4, 0x7}, // 2
    {0x7, 0x1, 0x7, 0x1, 0x7}, // 3
    {0x5, 0x5, 0x7, 0x1, 0x1}, // 4
    {0x7, 0x4, 0x7, 0x1, 0x7}, // 5
    {0x7, 0x4, 0x7, 0x5, 0x7}, // 6
    {0x7, 0x1, 0x1, 0x1, 0x1}, // 7
    {0x7, 0x5, 0x7, 0x5, 0x7}, // 8
    {0x7, 0x5, 0x7, 0x1, 0x7}, // 9
};

static void drawDigit(SDL_Renderer *r, int d, int x, int y, int s)
{
    for (int row = 0; row < 5; row++)
        for (int col = 0; col < 3; col++)
            if (kDigitFont[d % 10][row] & (1 << (2 - col)))
            {
                SDL_Rect rc{x + col * s, y + row * s, s, s};
                SDL_RenderFillRect(r, &rc);
            }
}

int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return 1;

    const int W = 1280, H = 720;
    SDL_Window *win = SDL_CreateWindow("Gamescope FrameGen Order Test",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    uint32_t frame = 0;
    bool running = true;
    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
                running = false;
        }

        // Dark gray background with faint vertical ruler lines every 64px
        SDL_SetRenderDrawColor(ren, 20, 22, 26, 255);
        SDL_RenderClear(ren);
        SDL_SetRenderDrawColor(ren, 55, 60, 70, 255);
        for (int gx = 0; gx < W; gx += 64)
            SDL_RenderDrawLine(ren, gx, 0, gx, H);

        // High-contrast moving vertical bar (steps +16px per real frame -> interpolated midpoint sits at +8px)
        int barX = int((frame * 16u) % (W - 40));
        SDL_SetRenderDrawColor(ren, 0, 235, 190, 255);
        SDL_Rect bar{barX, 160, 36, H - 220};
        SDL_RenderFillRect(ren, &bar);

        // Static foreground occluding pillar (tests occlusion/disocclusion edge handling!)
        SDL_SetRenderDrawColor(ren, 190, 60, 75, 255);
        SDL_Rect pillar{W / 2 - 40, 140, 80, H - 180};
        SDL_RenderFillRect(ren, &pillar);

        // Moving white square on opposite track for quick eye/camera check
        int sqX = int((frame * 24u) % (W - 60));
        SDL_SetRenderDrawColor(ren, 255, 220, 60, 255);
        SDL_Rect sq{sqX, 80, 56, 56};
        SDL_RenderFillRect(ren, &sq);

        // Big 4-digit real frame counter (top-left)
        SDL_SetRenderDrawColor(ren, 245, 245, 245, 255);
        uint32_t n = frame % 10000u;
        drawDigit(ren, (n / 1000) % 10, 40,  24, 8);
        drawDigit(ren, (n / 100)  % 10, 74,  24, 8);
        drawDigit(ren, (n / 10)   % 10, 108, 24, 8);
        drawDigit(ren,  n         % 10, 142, 24, 8);

        SDL_RenderPresent(ren);
        frame++;
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
