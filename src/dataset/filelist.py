with open('filelist.txt','w') as f:
    for i in range(2999):
        video_path=f"./first_frame/sample_{i:03d}/video_000.png"
        depth_path=f"./first_frame/sample_{i:03d}/depth_000.png"
        mask_path=f"./first_frame/sample_{i:03d}/mask_000.png"
        prompt_path=f"./first_frame/sample_{i:03d}/prompt_000.txt"
        hand_path=f"./hands/sample_{i:03d}/hand.jpg"
        normal_path=f"./normals/sample_{i:03d}/normal.jpg"
        f.write(f"{video_path} {mask_path} {depth_path} {prompt_path} {hand_path} {normal_path}\n")