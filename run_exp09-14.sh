echo "---------- Start: AMD_exp09_FOSI_Model_UNets_NPPCast.py ----------"
python AMD_exp09_FOSI_Model_UNets_NPPCast.py > AMD_exp09_FOSI_Model_UNets_NPPCast.log 2>&1
echo "AMD_exp09_FOSI_Model_UNets_NPPCast.py done. Log saved to AMD_exp09_FOSI_Model_UNets_NPPCast.log"

echo "---------- Start: AMD_exp10_OBS_input_FOSI_Model_Directly_NPPCast.py ----------"
python AMD_exp10_OBS_input_FOSI_Model_Directly_NPPCast.py > AMD_exp10_OBS_input_FOSI_Model_Directly_NPPCast.log 2>&1
echo "AMD_exp10_OBS_input_FOSI_Model_Directly_NPPCast.py done. Log saved to AMD_exp10_OBS_input_FOSI_Model_Directly_NPPCast.log"

echo "---------- Start: AMD_exp11_OBS_input_FOSI_Model_FineTune_NPPCast.py ----------"
python AMD_exp11_OBS_input_FOSI_Model_FineTune_NPPCast.py > AMD_exp11_OBS_input_FOSI_Model_FineTune_NPPCast.log 2>&1
echo "AMD_exp11_OBS_input_FOSI_Model_FineTune_NPPCast.py done. Log saved to AMD_exp11_OBS_input_FOSI_Model_FineTune_NPPCast.log"

echo "---------- Start: AMD_exp12_GIAF_Model_UNets_NPPCast.py ----------"
python AMD_exp12_GIAF_Model_UNets_NPPCast.py > AMD_exp12_GIAF_Model_UNets_NPPCast.log 2>&1
echo "AMD_exp12_GIAF_Model_UNets_NPPCast.py done. Log saved to AMD_exp12_GIAF_Model_UNets_NPPCast.log"

echo "---------- Start: AMD_exp13_input_GIAF_Model_Directly_NPPCast.py ----------"
python AMD_exp13_input_GIAF_Model_Directly_NPPCast.py > AMD_exp13_input_GIAF_Model_Directly_NPPCast.log 2>&1
echo "AMD_exp13_input_GIAF_Model_Directly_NPPCast.py done. Log saved to AMD_exp13_input_GIAF_Model_Directly_NPPCast.log"

echo "---------- Start: AMD_exp14_OBS_input_GIAF_Model_FineTune_NPPCast.py ----------"
python AMD_exp14_OBS_input_GIAF_Model_FineTune_NPPCast.py > AMD_exp14_OBS_input_GIAF_Model_FineTune_NPPCast.log 2>&1
echo "AMD_exp14_OBS_input_GIAF_Model_FineTune_NPPCast.py done. Log saved to AMD_exp14_OBS_input_GIAF_Model_FineTune_NPPCast.log"


echo "All experiments are finished."
