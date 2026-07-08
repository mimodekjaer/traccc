/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::device {

/// Interface shared by all device seeding algorithms
///
/// Every device seeding algorithm (triplet, GBTS) implements this same
/// interface, allowing clients to use them interchangeably.
///
/// The algorithms return a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying the buffer.
///
using seeding_algorithm = algorithm<edm::seed_collection::buffer(
    const edm::spacepoint_collection::const_view&,
    const edm::measurement_collection::const_view&)>;

}  // namespace traccc::device
