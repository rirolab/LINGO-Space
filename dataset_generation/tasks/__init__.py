from dataset_generation.tasks.lingo_space_composite import LingoSpaceComposite

from dataset_generation.tasks.spatial_relations import (
    AboveSeenColors, AboveUnseenColors, BelowSeenColors, \
    BelowUnseenColors, LeftSeenColors, LeftUnseenColors, \
    RightSeenColors, RightUnseenColors, FarSeenColors, 
    FarSeenColors, FarUnseenColors, CloseSeenColors, CloseUnseenColors
)
from dataset_generation.tasks.compositional_relations import (
    CompositionalRelationsSeenColors, CompositionalRelationsUnSeenColors
)

from dataset_generation.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from dataset_generation.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from dataset_generation.tasks.packing_shapes import PackingShapes


names = {
    # srem
    'left-seen-colors': LeftSeenColors,
    'left-unseen-colors': LeftUnseenColors,
    'right-seen-colors': RightSeenColors,
    'right-unseen-colors': RightUnseenColors,
    'behind-seen-colors': AboveSeenColors,
    'behind-unseen-colors': AboveUnseenColors,
    'front-seen-colors': BelowSeenColors,
    'front-unseen-colors': BelowUnseenColors,
    # Composition of relations
    'comp-one-step-seen-colors': CompositionalRelationsSeenColors,
    'comp-one-step-unseen-colors': CompositionalRelationsUnSeenColors,

    # cliport
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-shapes': PackingShapes,

    # our
    'far-seen-colors': FarSeenColors,
    'far-unseen-colors': FarUnseenColors,
    'close-seen-colors': CloseSeenColors,
    'close-unseen-colors': CloseUnseenColors,
    'composite': LingoSpaceComposite,
}
