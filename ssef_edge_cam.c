#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define CAM_W 640
#define CAM_H 480
#define CAM_BYTES (CAM_W * CAM_H * 3)
#define BMP_HDR_SIZE 54
#define BMP_TOTAL_SIZE (BMP_HDR_SIZE + CAM_BYTES)

static volatile int g_running = 1;
static void on_sigint(int sig) { (void)sig; g_running = 0; }

static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static uint8_t g_bmp[BMP_TOTAL_SIZE];
static uint64_t g_seq = 0;

static void init_bmp_header(uint8_t *hdr, int w, int h) {
    memset(hdr, 0, BMP_HDR_SIZE);
    uint32_t img_sz = (uint32_t)(w * h * 3);
    uint32_t file_sz = BMP_HDR_SIZE + img_sz;
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &file_sz, 4);
    uint32_t off = BMP_HDR_SIZE;
    memcpy(hdr + 10, &off, 4);
    uint32_t dib = 40;
    memcpy(hdr + 14, &dib, 4);
    int32_t bw = w, bh = -h;
    memcpy(hdr + 18, &bw, 4);
    memcpy(hdr + 22, &bh, 4);
    uint16_t planes = 1, bpp = 24;
    memcpy(hdr + 26, &planes, 2);
    memcpy(hdr + 28, &bpp, 2);
    memcpy(hdr + 34, &img_sz, 4);
}

static const char *mime_for(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return "application/octet-stream";
    if (strcmp(ext, ".html") == 0) return "text/html";
    if (strcmp(ext, ".js") == 0 || strcmp(ext, ".mjs") == 0) return "application/javascript";
    if (strcmp(ext, ".wasm") == 0) return "application/wasm";
    if (strcmp(ext, ".onnx") == 0 || strcmp(ext, ".task") == 0) return "application/octet-stream";
    return "application/octet-stream";
}

static void serve_static_file(int cli, const char *url_path) {
    char clean[256] = {0};
    const char *q = strchr(url_path, '?');
    size_t len = q ? (size_t)(q - url_path) : strlen(url_path);
    if (len >= sizeof(clean)) len = sizeof(clean) - 1;
    memcpy(clean, url_path, len);

    if (strcmp(clean, "/") == 0) strcpy(clean, "/index.html");
    if (strstr(clean, "..")) { close(cli); return; }

    char fpath[512];
    snprintf(fpath, sizeof(fpath), "/userdata/www%s", clean);
    FILE *f = fopen(fpath, "rb");
    if (!f) {
        const char *nf = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
        send(cli, nf, strlen(nf), MSG_NOSIGNAL);
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    char hdr[256];
    int hn = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %ld\r\nCache-Control: no-store\r\n\r\n",
        mime_for(fpath), sz);
    send(cli, hdr, (size_t)hn, MSG_NOSIGNAL);

    uint8_t buf[16384];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        if (send(cli, buf, n, MSG_NOSIGNAL) <= 0) break;
    }
    fclose(f);
}

static void *client_thread(void *arg) {
    int cli = (int)(intptr_t)arg;
    char req[1024] = {0};
    if (recv(cli, req, sizeof(req) - 1, 0) <= 0) { close(cli); return NULL; }

    char method[16] = {0}, path[256] = {0};
    sscanf(req, "%15s %255s", method, path);

    if (strcmp(path, "/stream") == 0) {
        const char *hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache\r\n\r\n";
        if (send(cli, hdr, strlen(hdr), MSG_NOSIGNAL) > 0) {
            uint8_t *local_bmp = malloc(BMP_TOTAL_SIZE);
            uint64_t last_seq = 0;
            while (g_running && local_bmp) {
                pthread_mutex_lock(&g_mtx);
                uint64_t cur = g_seq;
                if (cur != last_seq) memcpy(local_bmp, g_bmp, BMP_TOTAL_SIZE);
                pthread_mutex_unlock(&g_mtx);

                if (cur == last_seq) { usleep(10000); continue; }
                last_seq = cur;

                char part[128];
                int pn = snprintf(part, sizeof(part),
                    "--frame\r\nContent-Type: image/bmp\r\nContent-Length: %d\r\n\r\n",
                    BMP_TOTAL_SIZE);
                if (send(cli, part, (size_t)pn, MSG_NOSIGNAL) <= 0) break;
                if (send(cli, local_bmp, BMP_TOTAL_SIZE, MSG_NOSIGNAL) <= 0) break;
                if (send(cli, "\r\n", 2, MSG_NOSIGNAL) <= 0) break;
            }
            free(local_bmp);
        }
    } else {
        serve_static_file(cli, path);
    }
    close(cli);
    return NULL;
}

int main(void) {
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    init_bmp_header(g_bmp, CAM_W, CAM_H);

    void *cam_lib = dlopen("/userdata/libssef_capture_rgb.so", RTLD_NOW);
    if (!cam_lib) return 1;
    int (*ssef_rgb_init)(void) = dlsym(cam_lib, "ssef_rgb_init");
    int (*ssef_rgb_get_frame)(uint8_t *) = dlsym(cam_lib, "ssef_rgb_get_frame");
    void (*ssef_rgb_close)(void) = dlsym(cam_lib, "ssef_rgb_close");

    if (ssef_rgb_init() != 0) return 1;

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);
    bind(srv, (struct sockaddr *)&addr, sizeof(addr));
    listen(srv, 8);
    printf("[HYBRID EDGE] Ready on http://172.32.0.93:8080\n");

    uint8_t *cam_buf = malloc(CAM_BYTES);
    while (g_running) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(srv, &rfds);
        struct timeval tv = {0, 0};
        if (select(srv + 1, &rfds, NULL, NULL, &tv) > 0) {
            int cli = accept(srv, NULL, NULL);
            if (cli >= 0) {
                pthread_t tid;
                pthread_create(&tid, NULL, client_thread, (void *)(intptr_t)cli);
                pthread_detach(tid);
            }
        }

        if (ssef_rgb_get_frame(cam_buf) == CAM_BYTES) {
            pthread_mutex_lock(&g_mtx);
            uint8_t *dst = g_bmp + BMP_HDR_SIZE;
            for (int i = 0; i < CAM_W * CAM_H; i++) {
                dst[i * 3 + 0] = cam_buf[i * 3 + 2];
                dst[i * 3 + 1] = cam_buf[i * 3 + 1];
                dst[i * 3 + 2] = cam_buf[i * 3 + 0];
            }
            g_seq++;
            pthread_mutex_unlock(&g_mtx);
        }
    }

    ssef_rgb_close();
    free(cam_buf);
    close(srv);
    return 0;
}
