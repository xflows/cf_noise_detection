from django.shortcuts import render

def eval_to_table_view(request,input_dict,output_dict,widget):
    return render(request, 'visualizations/eval_to_table.html',{'widget':widget,'input_dict':input_dict,'output_dict':output_dict})
