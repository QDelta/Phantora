#pragma once

#define _GNU_SOURCE

#include <stdio.h>

void
exit(int);

#define NOT_IMPLEMENTED                                                        \
    do {                                                                       \
        fprintf(stderr, "NOT IMPLEMENTED: \"%s\"\n", __func__);                \
        exit(1);                                                               \
    } while (0)

struct phantora_cudaStream
{
    int device;
    int id;
};

struct phantora_cudaEvent
{
    int device;
    int stream;
    int id;
    long finished_time;
};

int
_get_current_device();

inline struct phantora_cudaStream
phantora_cudaStream(void* stream)
{
    struct phantora_cudaStream ret;
    if (stream) {
        struct phantora_cudaStream* stream_ = stream;
        ret.device = stream_->device;
        ret.id = stream_->id;
    } else {
        ret.device = _get_current_device();
        ret.id = 0;
    }
    return ret;
}
