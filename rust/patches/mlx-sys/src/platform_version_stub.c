/* Stub for __isPlatformVersionAtLeast for older SDKs */
#include <stdbool.h>
#include <stdint.h>
#include <AvailabilityVersions.h>

#ifdef __APPLE__
#include <TargetConditionals.h>

/* Provide weak symbol for __isPlatformVersionAtLeast if not available */
__attribute__((weak))
int32_t __isPlatformVersionAtLeast(uint32_t platform __attribute__((unused)),
                                    uint32_t major __attribute__((unused)),
                                    uint32_t minor __attribute__((unused)),
                                    uint32_t subminor __attribute__((unused))) {
    /* For macOS 14.0+, we can safely return true */
    return 1;
}
#endif

