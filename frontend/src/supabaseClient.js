import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://mjfurnmzgswteoxaniiu.supabase.co'
const supabaseKey = 'sb_publishable_mXQtqh9FY7sm3BmQ6Whjrg_F6gavV_M'

export const supabase = createClient(supabaseUrl, supabaseKey)
