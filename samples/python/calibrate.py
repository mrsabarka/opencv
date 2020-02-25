#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from common import splitfn

# built-in modules
import os


def undistortedfunction(img_names, debug_dir, camera_matrix, dist_coefs):
    for fn in img_names if debug_dir else []:
        _path, name, _ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_chess.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0, (w, h))

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    return newcameramtx, roi, dst


def main():
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)
    args.setdefault('--leftimages', '../data/a/left??.png')
    args.setdefault('--rightimages', '../data/a/right??.png')
    img_mask = args.get('--leftimages')
    img_mask2 = args.get('--rightimages')

    img_names = glob(img_mask)
    img_names2 = glob(img_mask2)
    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))

    pattern_size = (8, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    obj_points2 = []
    img_points2 = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

    def rectifiedImage(img_names, debug_dir, map1, map2):
        for fn in img_names if debug_dir else []:
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_rectified.png')

            img = cv.imread(fn)
            dst = cv.remap(img, map1, map2, cv.INTER_NEAREST)
            print('rectified image written to: %s' % outfile)
            cv.imwrite(outfile, dst)

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % fn)
        return (corners.reshape(-1, 2), pattern_points)

    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
        chessboards2 = [processImage(fn) for fn in img_names2]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)
        chessboards2 = pool.map(processImage, img_names2)

    chessboards = [x for x in chessboards if x is not None]
    chessboards2 = [x for x in chessboards2 if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
    for (corners, pattern_points) in chessboards2:
        img_points2.append(corners)
        obj_points2.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    rms2, camera_matrix2, dist_coefs2, _rvecs2, _tvecs2 = cv.calibrateCamera(obj_points2, img_points2, (w, h), None,
                                                                             None)

    flags = cv.CALIB_FIX_INTRINSIC

    # stereo calibration
    retval, newcamera_matrix, newdistCoeffs, newcamera_matrix2, newdistCoeffs2, R, T, E, F_matrix = cv.stereoCalibrate(
        obj_points, img_points, img_points2, camera_matrix, dist_coefs, camera_matrix2, dist_coefs2, imageSize=(w, h),
        flags=flags)

    # stereo rectify
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(newcamera_matrix, newdistCoeffs, newcamera_matrix2,
                                                                     newdistCoeffs2, (w, h), R, T)

    # undistort the image with the calibration
    undistortedfunction(img_names, debug_dir, camera_matrix, dist_coefs)
    undistortedfunction(img_names2, debug_dir, camera_matrix2, dist_coefs2)

    map1, map2 = cv.initUndistortRectifyMap(newcamera_matrix, dist_coefs, R2, P2, (w, h), cv.CV_32FC1)
    map3, map4 = cv.initUndistortRectifyMap(newcamera_matrix2, dist_coefs2, R2, P2, (w, h), cv.CV_32FC1)

    rectifiedImage(img_names, debug_dir, map1, map2)
    rectifiedImage(img_names2, debug_dir, map3, map4)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
