// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   sdkl_compat.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  The one place nntrainer includes the HexKL SDK headers from.
 *
 * sdkl.h spells the C99 `restrict` keyword in its prototypes. C++ has no
 * such keyword, so the NDK C++ front-end rejects the header as written.
 * Mapping it onto the compiler extension for the duration of this include --
 * and undefining it immediately after -- keeps `restrict` an ordinary
 * identifier everywhere else, which a project-wide -Drestrict would not:
 * any variable or member named `restrict` anywhere in the tree would then
 * be silently rewritten.
 *
 * sdkl.h also references CDSP_DOMAIN_ID / CDSP1_DOMAIN_ID from remote.h, so
 * remote.h has to come first; including both here means no caller has to
 * remember the order.
 */

#ifndef __SDKL_COMPAT_H__
#define __SDKL_COMPAT_H__

#ifdef ENABLE_HEXKL

#include <remote.h>

#ifdef restrict
#error "sdkl_compat.h: `restrict` is already defined; refusing to shadow it"
#endif

#define restrict __restrict
#include <sdkl.h>
#undef restrict

#endif // ENABLE_HEXKL

#endif // __SDKL_COMPAT_H__
