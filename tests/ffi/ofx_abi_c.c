#include <stddef.h>
#include <stdint.h>

#include "ofx_abi_types.h"

#define FJ_OFX_TYPE(type, size, alignment)                \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type);
#define FJ_OFX_FIELD(type, field, offset, expected)                               \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type "." #field); \
    _Static_assert(_Generic(((type*)0)->field, expected: 1, default: 0), "type: " #type "." #field);
#define FJ_OFX_VALUE(name, expected) _Static_assert(name == expected, "value: " #name);
#include "ofx_abi_facts.inc"
#undef FJ_OFX_VALUE
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
_Static_assert(_Generic((OfxStatus)0, int: 1, default: 0), "signed status");
_Static_assert(_Generic((OfxTime)0, double: 1, default: 0), "time");
_Static_assert(_Generic((OfxPropertySetHandle)0, struct OfxPropertySetStruct*: 1, default: 0), "property handle");
_Static_assert(_Generic((OfxParamHandle)0, struct OfxParamStruct*: 1, default: 0), "parameter handle");
_Static_assert(_Generic((OfxParamSetHandle)0, struct OfxParamSetStruct*: 1, default: 0), "parameter set handle");
_Static_assert(_Generic((OfxImageEffectHandle)0, struct OfxImageEffectStruct*: 1, default: 0), "effect handle");
_Static_assert(_Generic((OfxImageClipHandle)0, struct OfxImageClipStruct*: 1, default: 0), "clip handle");
_Static_assert(_Generic((OfxImageMemoryHandle)0, struct OfxImageMemoryStruct*: 1, default: 0), "memory handle");

int main(void) {
    return 0;
}
