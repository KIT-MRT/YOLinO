RUN_ID=$1
echo $RUN_ID
mkdir debug/prediction/$RUN_ID/clips/0531/movies/

# produce
echo " ------------------ create video from images ------------------ "
find debug/prediction/$RUN_ID/clips/0531 -maxdepth 1 -name "149*" -type d -exec python ../yolino/src/yolino/tools/make_videos.py --height 1500 -o debug/prediction/$RUN_ID/clips/0531/movies/ -i {} --pattern .jpg_3_pred_spline.png \;

# combine
echo " ------------------ combine single sequences to a single video ------------------ "
find "$(pwd)"/debug/prediction/$RUN_ID/clips/0531/movies/ -name "*3predspline.mp4" -exec echo "file {}" \; > /tmp/movies.txt
ffmpeg -f concat -safe 0 -i /tmp/movies.txt -c copy debug/prediction/$RUN_ID/clips/0531/movies/combined_spline.mp4 -y 2> errors.txt

# add title
echo " ------------------ add title ------------------ "
ffmpeg -i debug/prediction/$RUN_ID/clips/0531/movies/combined_spline.mp4 -i title.png -filter_complex "[0:v][1:v] overlay=0:0:enable='between(t,0,20)'" -pix_fmt yuv420p -c:a copy debug/prediction/$RUN_ID/clips/0531/movies/combined_title_spline.mp4 -y 2>> errors.txt

echo "vlc" $(pwd)/debug/prediction/$RUN_ID/clips/0531/movies/combined_title_spline.mp4

# ----------------------------------------------------------------

echo $RUN_ID
mkdir debug/prediction/$RUN_ID/clips/0531/movies/

# produce
echo " ------------------ create video from images ------------------ "
find debug/prediction/$RUN_ID/clips/0531 -maxdepth 1 -name "149*" -type d -exec python ../yolino/src/yolino/tools/make_videos.py --height 1500 -o debug/prediction/$RUN_ID/clips/0531/movies/ -i {} --pattern .jpg_4_nms_.png \;

# combine
echo " ------------------ combine single sequences to a single video ------------------ "
find "$(pwd)"/debug/prediction/$RUN_ID/clips/0531/movies/ -name "*4nms.mp4" -exec echo " ------------------file {}" \; > /tmp/movies.txt
ffmpeg -f concat -safe 0 -i /tmp/movies.txt -c copy debug/prediction/$RUN_ID/clips/0531/movies/combined_nms.mp4 -y 2>> errors.txt

# add title
echo " ------------------ add title ------------------ "
ffmpeg -i debug/prediction/$RUN_ID/clips/0531/movies/combined_nms.mp4 -i title.png -filter_complex "[0:v][1:v] overlay=0:0:enable='between(t,0,20)'" -pix_fmt yuv420p -c:a copy debug/prediction/$RUN_ID/clips/0531/movies/combined_title_nms.mp4 -y 2>> errors.txt

echo "vlc" $(pwd)/debug/prediction/$RUN_ID/clips/0531/movies/combined_title_nms.mp4
