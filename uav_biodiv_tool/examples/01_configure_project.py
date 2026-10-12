from uav_biodiv import UAVProjectConfig


def main() -> None:
    cfg = UAVProjectConfig(
        project_name="BioClima",
        base_data_dir=r"D:\BioClima",
        crs="EPSG:3844",
    )

    # v0.5.1 default: hardware-adaptive processing. No workstation-specific
    # RAM, CPU, chunk or tile sizes are required.
    cfg.set_processing_limits(mode="auto")
    print("Automatic resource plan:")
    for key, value in cfg.processing_limits.resource_plan().items():
        print(f"  {key}: {value}")

    # All sensor and channel definitions are supplied by the user. The package
    # itself has no manufacturer/sensor/channel presets.
    cfg.add_sensor(
        "sensor_A",
        sensor_type="multiband",
        bands={
            "channel_alpha": {
                "role": "feature_alpha",
                "raster_band": 1,
                "point_attribute": "SignalAlpha",
            },
            "channel_beta": {
                "role": "feature_beta",
                "raster_band": 2,
                "point_attribute": "SignalBeta",
            },
        },
        indices={
            "index_1": {
                "method": "normalized_difference",
                "a_role": "feature_beta",
                "b_role": "feature_alpha",
            }
        },
    )

    cfg.add_sensor(
        "point_sensor_A",
        sensor_type="point_cloud",
        # Examples only. Use actual dimensions reported by inspect_point_cloud().
        point_attributes={
            "intensity": "MyAmplitudeDimension",
            "classification": "MyClassDimension",
            "height": "MyNormalizedHeightDimension",
        },
        ground_classes=[2],
    )

    cfg.save_json("bioclima_config.json")
    print("Saved bioclima_config.json")


if __name__ == "__main__":
    main()
