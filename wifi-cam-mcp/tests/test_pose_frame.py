"""Left has to read as left.

The device reports pan with positive x meaning physically left -- `_move_impl`
relies on it, sending +x for Direction.LEFT. `get_hw_position` passed that value
through untouched while flipping tilt, so the pose it returned was half in the
user's frame and half in the device's.

Nothing consumed it yet, so nothing was visibly wrong. But `body_contingency`
already declares the other convention -- `look_left` expects pan -1.0 and
`look_right` expects +1.0 -- so the first person to wire the two together, which
is the obvious thing to do when improving spatial search, would have inverted
every camera agency verdict at once.

One frame for both axes: positive pan is right, positive tilt is up, from where
the person sitting in the room stands.
"""

from __future__ import annotations

from wifi_cam_mcp.camera import CameraPosition, describe_pose, to_user_frame


class TestPanReadsInTheFrameItIsCommandedIn:
    def test_device_positive_x_is_physical_left_so_it_reads_negative(self) -> None:
        assert to_user_frame(0.4, 0.0, "normal").pan < 0

    def test_device_negative_x_is_physical_right_so_it_reads_positive(self) -> None:
        assert to_user_frame(-0.4, 0.0, "normal").pan > 0

    def test_it_matches_what_body_contingency_already_expects(self) -> None:
        # look_left -> ('pan', -1.0), look_right -> ('pan', +1.0).
        looking_left = to_user_frame(0.4, 0.0, "normal").pan
        looking_right = to_user_frame(-0.4, 0.0, "normal").pan

        assert looking_left < 0 < looking_right


class TestTiltKeepsTheConventionItAlreadyHad:
    def test_device_positive_y_is_physical_down_so_it_reads_negative(self) -> None:
        assert to_user_frame(0.0, 0.5, "normal").tilt < 0

    def test_device_negative_y_is_physical_up_so_it_reads_positive(self) -> None:
        assert to_user_frame(0.0, -0.5, "normal").tilt > 0


class TestCeilingMountMirrorsBothAxes:
    def test_pan_mirrors(self) -> None:
        # Upside-down: what the device calls left is on the person's right.
        assert to_user_frame(0.4, 0.0, "ceiling").pan == -to_user_frame(
            0.4, 0.0, "normal"
        ).pan

    def test_tilt_mirrors(self) -> None:
        assert to_user_frame(0.0, 0.5, "ceiling").tilt == -to_user_frame(
            0.0, 0.5, "normal"
        ).tilt


class TestTheDescriptionSaysTheDirectionOutLoud:
    def test_a_negative_pan_is_called_left(self) -> None:
        # The word is the check on the sign. A mirrored axis shows up as a
        # description that contradicts the picture, which a person notices;
        # a bare -0.40 does not.
        assert "left" in describe_pose(CameraPosition(pan=-0.4, tilt=0.0))

    def test_a_positive_pan_is_called_right(self) -> None:
        assert "right" in describe_pose(CameraPosition(pan=0.4, tilt=0.0))

    def test_it_reports_degrees_not_normalized_units(self) -> None:
        # 0.4 of a 180 degree range is 72 degrees.
        assert "72" in describe_pose(CameraPosition(pan=-0.4, tilt=0.0))

    def test_it_names_the_point_the_angle_is_measured_from(self) -> None:
        # Without this the number reads as a fact about the room rather than
        # about the camera's own travel, and a heading remembered in one frame
        # gets acted on in the other.
        assert "from centre" in describe_pose(CameraPosition(pan=-0.4, tilt=0.0))

    def test_a_centred_camera_says_so_instead_of_zero_degrees_of_nothing(self) -> None:
        assert describe_pose(CameraPosition(pan=0.0, tilt=0.0)) == "aimed at centre"

    def test_one_axis_off_centre_mentions_only_that_axis(self) -> None:
        described = describe_pose(CameraPosition(pan=0.0, tilt=0.5))

        assert "up" in described
        assert "left" not in described
        assert "right" not in described

    def test_an_unreported_pose_says_so_rather_than_guessing(self) -> None:
        described = describe_pose(None)

        assert "unknown" in described
        assert "deg" not in described


class TestCentreIsCentreEitherWay:
    def test_zero_stays_zero(self) -> None:
        # A mount-mode bug that only shows up off-centre is worth ruling out.
        for mount in ("normal", "ceiling"):
            pose = to_user_frame(0.0, 0.0, mount)
            assert pose.pan == 0.0
            assert pose.tilt == 0.0
