.\gan_venv\Scripts\Activate.ps1
$visdomProcess = Start-Process -FilePath "python.exe" -ArgumentList "-m visdom.server -p 8097" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3
python .\train.py --continue_train --epoch_count 2 --n_epochs 14 --lr_policy "cosine" --batch_size 32 --dataroot "G:/cv_report/diffusionmodel/img_align_celeba/" --gan_mode "vanilla" --name "gan_colorization" --model "colorization" --dataset_mode "colorization" --input_nc 1 --output_nc 2 --crop_size 256
Stop-Process -Id $visdomProcess.Id
deactivate