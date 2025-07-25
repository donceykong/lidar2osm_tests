// ************************ KITTI DATA HASH MAPS ************************* //

// static std::map<int, std::string> labelStringMap = {
    // {0, "unlabeled"},
    // {1, "outlier"},
    // {0, "unlabeled"},
    // {1, "outlier"},
    // {10, "car"},
    // {11, "bicycle"},
    // {13, "bus"},
    // {15, "motorcycle"},
    // {16, "on-rails"},
    // {18, "truck"},
    // {20, "other-vehicle"},
    // {30, "person"},
    // {31, "bicyclist"},
    // {32, "motorcyclist"},
    // {40, "road"},
    // {44, "parking"},
    // {48, "sidewalk"},
    // {49, "other-ground"},
    // {50, "building"},
    // {51, "fence"},
    // {52, "other-structure"},
    // {60, "lane-marking"},
    // {70, "vegetation"},
    // {71, "trunk"},
    // {72, "terrain"},
    // {80, "pole"},
    // {81, "traffic-sign"},
    // {99, "other-object"},
    // {252, "moving-car"},
    // {256, "moving-bicyclist"},
    // {253, "moving-person"},
    // {254, "moving-motorcyclist"},
    // {255, "moving-on-rails"},
    // {257, "moving-bus"},
    // {258, "moving-truck"},
    // {259, "moving-other-vehicle"}
// };

// static std::map<int, std::vector<int>> color_map = {
    // {45, "unlabeled", {255, 0, 0}},
    // {46, "unlabeled", {0, 0, 255}}
    // {0, {0, 0, 0}},
    // {1, {0, 0, 255}},
    // {10, {245, 150, 100}},
    // {11, {245, 230, 100}},
    // {13, {250, 80, 100}},
    // {15, {150, 60, 30}},
    // {16, {255, 0, 0}},
    // {18, {180, 30, 80}},
    // {20, {255, 0, 0}},
    // {30, {30, 30, 255}},
    // {31, {200, 40, 255}},
    // {32, {90, 30, 150}},
    // {40, {255, 0, 255}},
    // {44, {255, 150, 255}},
    // {48, {75, 0, 75}},
    // {49, {75, 0, 175}},
    // {50, {0, 200, 255}},
    // {51, {50, 120, 255}},
    // {52, {0, 150, 255}},
    // {60, {170, 255, 150}},
    // {70, {0, 175, 0}},
    // {71, {0, 60, 135}},
    // {72, {80, 240, 150}},
    // {80, {150, 240, 255}},
    // {81, {0, 0, 255}},
    // {99, {255, 255, 50}},
    // {252, {245, 150, 100}},
    // {256, {255, 0, 0}},
    // {253, {200, 40, 255}},
    // {254, {30, 30, 255}},
    // {255, {90, 30, 150}},
    // {257, {250, 80, 100}},
    // {258, {180, 30, 80}},
    // {259, {255, 0, 0}}
// };


// Define the map with tuple
static std::map<int, std::tuple<std::string, std::vector<int>>> label_map = {
    {45, {"osm_road", {255, 0, 0}}},
    {46, {"osm_building", {0, 0, 255}}},
    {10, {"road", {245, 150, 100}}},
};

int getLabelFromRGB(int r, int g, int b) {
    for (const auto& [label, info] : label_map) {
        const std::vector<int>& rgb = std::get<1>(info);  // RGB vector
        if (rgb[0] == r && rgb[1] == g && rgb[2] == b) {
            return label;
        }
    }
    return -1;
}

std::vector<int> getRGBFromLabel(int label) {
    auto it = label_map.find(label);
    if (it != label_map.end()) {
        return std::get<1>(it->second);  // Return RGB vector
    } else {
        return {-1, -1, -1};  // Indicate invalid label
    }
}